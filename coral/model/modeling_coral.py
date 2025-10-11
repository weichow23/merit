# -*- coding: utf-8 -*-
# Author: Yuan Gao, Linfeng Li, Xian Wang
# Date: 2025-09-20
# Project: MERIT
# Description: Experiement code for MERIT

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from transformers.file_utils import ModelOutput

from model.enhancedDecoder import BertLayerForDecoder


def split_dict_values(
    original_dict: Dict[str, Optional[Tensor]],
    has_hardneg: bool,
    is_qwenvl: bool = False,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Split a dictionary of batched tensors into query and passage parts.

    Behavior:
      - If has_hardneg is False:
          Split the batch dimension into 2 chunks: [query | passage].
      - If has_hardneg is True:
          Split the batch dimension into 3 chunks: [query | positive | hardneg],
          but only return query and positive (passage).

    Special handling for QwenVL position_ids (shape: [3, B, L]):
      - Split along the batch axis at dim=1.

    Skips:
      - None values.
      - 'past_key_values' (decoder cache).

    Args:
      original_dict: Dict of batched tensors to split.
      has_hardneg: Whether inputs are concatenated as (q, p, h).
      is_qwenvl: Uses special split for 3D position_ids: ([3, B, L]).

    Returns:
      (query_dict, passage_dict): Two dictionaries with split tensors.
    """
    query_dict: Dict[str, Tensor] = {}
    passage_dict: Dict[str, Tensor] = {}

    # Helper to split evenly with integer division; assumes inputs are correctly concatenated
    def _split_two(x: Tensor) -> Tuple[Tensor, Tensor]:
        mid = len(x) // 2
        return x[:mid], x[mid:]

    def _split_three_take_q_p(x: Tensor) -> Tuple[Tensor, Tensor]:
        third = len(x) // 3
        return x[:third], x[third : 2 * third]

    for key, value in original_dict.items():
        if value is None:
            continue
        if key == "past_key_values":
            # Skip decoder cache if present
            continue

        if key == "position_ids" and is_qwenvl:
            # QwenVL position_ids: [3, B, L], batch dim is dim=1
            if value.dim() != 3 or value.shape[0] != 3:
                raise ValueError(
                    f"Expected QwenVL position_ids shape [3, B, L], got {tuple(value.shape)}"
                )
            if not has_hardneg:
                mid = value.shape[1] // 2
                query_dict[key] = value[:, :mid, :]
                passage_dict[key] = value[:, mid:, :]
            else:
                third = value.shape[1] // 3
                query_dict[key] = value[:, :third, :]
                passage_dict[key] = value[:, third : 2 * third, :]
            continue

        # Default path: split along batch dimension (dim=0)
        if not has_hardneg:
            qv, pv = _split_two(value)
        else:
            qv, pv = _split_three_take_q_p(value)

        query_dict[key] = qv
        passage_dict[key] = pv

    return query_dict, passage_dict


class DualDecoder(nn.Module):
    """
    A lightweight dual cross-attention decoder:
      - language_decoder attends to encoder states with a language mask
      - vision_decoder attends to encoder states with a vision mask
    """

    def __init__(self, config, dtype: torch.dtype):
        super().__init__()
        self.language_decoder = BertLayerForDecoder(config).to(dtype=dtype)
        self.vision_decoder = BertLayerForDecoder(config).to(dtype=dtype)

    def forward(
        self,
        query: Tensor,
        hiddens: Tensor,
        language_matrix_attention_mask: Tensor,
        vision_matrix_attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          query: [B, L, H] query embeddings (positional + EOS-based)
          hiddens: [B, L, H] encoder hidden states (KV)
          language_matrix_attention_mask: extended attention mask for language tokens
          vision_matrix_attention_mask: extended attention mask for vision tokens

        Returns:
          language_hiddens, vision_hiddens: the respective decoder outputs
        """
        language_hiddens = self.language_decoder(
            query=query,
            key=hiddens,
            value=hiddens,
            attention_mask=language_matrix_attention_mask.to(hiddens.dtype),
        )[0]

        vision_hiddens = self.vision_decoder(
            query=query,
            key=hiddens,
            value=hiddens,
            attention_mask=vision_matrix_attention_mask.to(hiddens.dtype),
        )[0]

        return language_hiddens, vision_hiddens


class BaseForCORAL(nn.Module):
    """
    CORAL base model with:
      - Single encoder (multimodal)
      - Dual decoders for masked language modeling and vision regression
      - Contrastive objective (in-batch and/or cross-device)
    """

    def __init__(self, encoder_model: nn.Module, config, **kwargs):
        super().__init__()
        self.encoder = encoder_model
        self.config = config

        # Ensure decoder stability
        self.config.hidden_dropout_prob = 0.1
        self.config.layer_norm_eps = 1e-5

        # Decoders
        self.q2p_decoder = DualDecoder(self.config, self.encoder.dtype)
        self.p2p_decoder = DualDecoder(self.config, self.encoder.dtype)

        # Positional embeddings used to form decoder queries
        self.decoder_position_embeddings = nn.Embedding(
            num_embeddings=self.config.max_position_embeddings,
            embedding_dim=self.config.hidden_size,
        ).to(dtype=self.encoder.dtype)

        # Vision regression head: projects decoder output back to hidden size
        self.vision_head = nn.Linear(config.hidden_size, config.hidden_size)

        # Losses
        # Use default reduction='mean' (reduce is deprecated)
        self.ce_loss_fnt = nn.CrossEntropyLoss()
        self.mse_loss_fnt = nn.MSELoss()

        # Training hyperparameters (from kwargs)
        self.config.l_mask_prob = kwargs["l_mask_prob"]
        self.config.v_mask_prob = kwargs["v_mask_prob"]
        self.config.lambda_l = kwargs["lambda_l"]
        self.config.lambda_v = kwargs["lambda_v"]
        self.config.select_embed = kwargs["select_embed"]
        self.config.contrastive_temp = kwargs["contrastive_temp"]
        self.config.negatives_cross_device = kwargs["negatives_cross_device"]
        self.config.use_inbatch_neg = kwargs["use_inbatch_neg"]
        self.has_hardneg = kwargs["has_hardneg"]

        # Cross-device negatives require initialized torch.distributed
        if self.config.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError(
                    "Distributed training has not been initialized for cross-device negatives."
                )
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def get_model(self) -> nn.Module:
        """Return the underlying encoder module."""
        return self.encoder

    def encode(self, features: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Encode inputs and return normalized pooled representations.

        Args:
          features: Input dict for the encoder.

        Returns:
          p_reps: [B, H] normalized representation (last token pooling).
          outputs: Full encoder outputs (with hidden_states).
        """
        outputs = self.encoder(**features, return_dict=True, output_hidden_states=True)
        # Pooling: take the last token representation
        p_reps = outputs.hidden_states[-1][:, -1, :]
        p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous(), outputs

    def generate_mm_token_mask(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        mlm_prob: float,
        mask_modality: str = "language",
    ) -> Tensor:
        """
        Generate a per-sample token masking matrix [B, L, L] for decoder attention.

        For each sample:
          - Start with Bernoulli(mlm_prob) across [L, L].
          - Zero out rows corresponding to PAD tokens.
          - Zero out diagonal (no self-attention masking noise).
          - Unmask special tokens and the opposite modality so they remain always visible.

        Special tokens:
          - Those with id >= bos_token_id (but not image_token_id) are considered special language tokens.
        Modality selection:
          - If mask_modality == "language": unmask vision/image tokens and special tokens.
          - Else (mask_modality == "vision"): unmask language tokens (<= bos_token_id).

        Returns:
          matrix_attention_mask: [B, L, L], 1=keep, 0=mask
        """
        B, L = attention_mask.shape
        device = attention_mask.device
        matrix_attention_mask = []

        for b in range(B):
            # Initial Bernoulli matrix
            probs = torch.full((L, L), mlm_prob, device=device)
            m = torch.bernoulli(probs)

            # Unmasking rules
            # 1) Keep rows for valid tokens (mask out PAD rows)
            m[attention_mask[b] == 0] = 0
            # 2) Do not mask self (decoder can always attend to itself baseline)
            m.fill_diagonal_(0)

            # Identify special and modality tokens
            is_special_lang = (
                (input_ids[b] >= self.config.bos_token_id)
                & (input_ids[b] != self.config.image_token_id)
                & (attention_mask[b] == 1)
            )
            if mask_modality == "language":
                # Unmask image tokens when masking language
                is_opposite_modality = input_ids[b] == self.config.image_token_id
            else:
                # Unmask language tokens when masking vision
                is_opposite_modality = input_ids[b] <= self.config.bos_token_id

            # Rows corresponding to special tokens or opposite modality are fully unmasked (set to 1)
            m[is_special_lang] = 1
            m[is_opposite_modality] = 1

            matrix_attention_mask.append(m)

        matrix_attention_mask = torch.stack(matrix_attention_mask, dim=0)
        return matrix_attention_mask

    def mlm_loss(self, hiddens: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute MLM cross-entropy over decoder outputs.

        Args:
          hiddens: [B, L, H] decoder outputs for language
          labels: [B, L] token ids

        Returns:
          logits: [B, L, V]
          loss: scalar CE loss
        """
        logits = self.encoder.lm_head(hiddens)
        loss = self.ce_loss_fnt(
            logits.view(-1, self.encoder.config.vocab_size),
            labels.view(-1),
        )
        return logits, loss

    def mse_loss(self, hiddens: Tensor, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Regress decoder vision outputs back to encoder input embeddings.

        Args:
          hiddens: [B, L, H] decoder outputs for vision
          features: [B, L, H] target embeddings (e.g., inputs_embeds)

        Returns:
          pred_visual: [B, L, H]
          loss: scalar MSE loss
        """
        pred_visual = self.vision_head(hiddens)
        loss = self.mse_loss_fnt(pred_visual, features)
        return pred_visual, loss

    def _decoder_forward_once(
        self,
        decoder: DualDecoder,
        eos: Tensor,
        inputs: Dict[str, Tensor],
        outputs: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Run a single decoder (q2p or p2p) pass to produce MLM and MSE losses.

        Args:
          decoder: DualDecoder module
          eos: [B, H] pooled representations used to shift/seed the decoder input
          inputs: encoder input dict (split)
          outputs: encoder output dict (split)

        Returns:
          (mlm_loss, mse_loss): per-branch losses
        """
        # Qwen2VL uses 3D position_ids: [3, B, L]; else it's [B, L]
        position_ids = outputs["position_ids"][0]  # idx=0 selects the "position index type"
        # Build decoder inputs: concatenate EOS as first token embedding
        # Replace first token in inputs_embeds with EOS, keep subsequent tokens
        hiddens = torch.cat([eos.unsqueeze(1), outputs["inputs_embeds"][:, 1:]], dim=1)

        # Positional query = learned position embeddings + EOS (per position)
        positional_embeddings = self.decoder_position_embeddings(position_ids)
        query = positional_embeddings + eos.unsqueeze(1)

        # Construct modality-aware masking matrices
        language_attention_mask = self.generate_mm_token_mask(
            inputs["input_ids"],
            outputs["attention_mask"],
            self.config.l_mask_prob,
            mask_modality="language",
        )
        vision_attention_mask = self.generate_mm_token_mask(
            inputs["input_ids"],
            outputs["attention_mask"],
            self.config.v_mask_prob,
            mask_modality="vision",
        )

        # Convert [B, L, L] masks to extended form accepted by the decoder attention
        language_matrix_attention_mask = self.encoder.get_extended_attention_mask(
            language_attention_mask,
            language_attention_mask.shape,
            language_attention_mask.device,
        )
        vision_matrix_attention_mask = self.encoder.get_extended_attention_mask(
            vision_attention_mask,
            vision_attention_mask.shape,
            vision_attention_mask.device,
        )

        # Dual decoding
        language_hiddens, vision_hiddens = decoder(
            query, hiddens, language_matrix_attention_mask, vision_matrix_attention_mask
        )

        # Losses
        _, mlm_loss = self.mlm_loss(language_hiddens, inputs["input_ids"])
        _, mse_loss = self.mse_loss(vision_hiddens, outputs["inputs_embeds"])

        return mlm_loss, mse_loss

    def forward(
        self,
        query: Optional[Dict[str, Tensor]] = None,
        passage: Optional[Dict[str, Tensor]] = None,
        passage_neg: Optional[Dict[str, Tensor]] = None,
    ) -> Union[Tuple, ModelOutput]:
        """
        Forward pass:
          - Encode concatenated inputs (q, p[, h]).
          - Split encoder inputs/outputs into q/p (and discard h for decoders).
          - Run q->p and p->p decoders to get MLM and MSE losses.
          - Optionally compute contrastive loss (in-batch / cross-device).

        Args:
          query: Not used directly (inputs are in 'passage' in current pipeline).
          passage: Dict with concatenated inputs (q, p[, h]).
          passage_neg: Optionally present; not used directly when 'has_hardneg' is True since negatives are concatenated.

        Returns:
          ModelOutput with:
            - loss: total loss (decoder + contrastive if training)
            - contrastive_loss, mae_loss, and detailed components
        """
        # Encode all concatenated inputs
        encoder_hiddens, encoder_outputs = self.encode(passage)

        # Split encoder inputs/outputs into q/p parts (handle Qwen2VL position_ids shape)
        q_in, p_in = split_dict_values(
            passage, self.has_hardneg, self.encoder.config.model_type == "qwen2_5_vl"
        )
        q_out, p_out = split_dict_values(
            encoder_outputs, self.has_hardneg, self.encoder.config.model_type == "qwen2_5_vl"
        )

        # Split pooled reps
        if not self.has_hardneg:
            half = len(encoder_hiddens) // 2
            q_reps = encoder_hiddens[:half]
            p_reps = encoder_hiddens[half:]
        else:
            third = len(encoder_hiddens) // 3
            q_reps = encoder_hiddens[:third]
            p_reps = encoder_hiddens[third : 2 * third]
            h_reps = encoder_hiddens[2 * third :]

        # Decoder passes
        q2p_mlm_loss, q2p_mse_loss = self._decoder_forward_once(
            decoder=self.q2p_decoder, eos=q_reps, inputs=p_in, outputs=p_out
        )
        p2p_mlm_loss, p2p_mse_loss = self._decoder_forward_once(
            decoder=self.p2p_decoder, eos=p_reps, inputs=p_in, outputs=p_out
        )

        # Weighted decoder losses
        q2p_loss = self.config.lambda_l * q2p_mlm_loss + self.config.lambda_v * q2p_mse_loss
        p2p_loss = self.config.lambda_l * p2p_mlm_loss + self.config.lambda_v * p2p_mse_loss

        # Contrastive loss
        if self.training and (self.config.contrastive_temp > 0.0):
            # Cross-device in-batch negatives
            if self.config.negatives_cross_device and self.config.use_inbatch_neg:
                if self.has_hardneg:
                    q_reps = self._dist_gather_tensor(q_reps)
                    p_reps = self._dist_gather_tensor(p_reps)
                    h_reps = self._dist_gather_tensor(h_reps)
                    # Append hard negatives to passages
                    p_reps = torch.cat([p_reps, h_reps], dim=0)
                else:
                    q_reps = self._dist_gather_tensor(q_reps)
                    p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            temp = self.config.contrastive_temp

            if self.config.use_inbatch_neg:
                # Full in-batch similarity: [B, B*G]
                scores = self.compute_similarity(q_reps, p_reps) / temp
                scores = scores.view(q_reps.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                loss_contrastive = self.ce_loss_fnt(scores, target)
            else:
                # Only positives per sample: group positives in contiguous chunks
                # p_reps shaped as [B*G, H], reshape to [B, G, H] and compute sim with q_reps [B, 1, H]
                scores = (
                    self.compute_similarity(
                        q_reps[:, None, :], p_reps.view(q_reps.size(0), group_size, -1)
                    ).squeeze(1)
                    / temp
                )  # [B, G]
                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss_contrastive = self.ce_loss_fnt(scores, target)

        elif self.training:
            # Only decoder losses (no contrastive)
            loss_contrastive = torch.tensor(0.0, device=q_reps.device, dtype=q_reps.dtype)
        else:
            # Evaluation: return similarity scores for retrieval
            scores = self.compute_similarity(q_reps, p_reps)
            loss_contrastive = None

        total_mae = q2p_loss + p2p_loss
        total_loss = total_mae + (loss_contrastive if loss_contrastive is not None else 0.0)

        return ModelOutput(
            loss=total_loss,
            contrastive_loss=loss_contrastive,
            mae_loss=total_mae,
            q2p_loss=q2p_loss,
            p2p_loss=p2p_loss,
            mlm_loss=q2p_mlm_loss + p2p_mlm_loss,
            mse_loss=q2p_mse_loss + p2p_mse_loss,
            q2p_mlm_loss=q2p_mlm_loss,
            p2p_mlm_loss=p2p_mlm_loss,
            q2p_mse_loss=q2p_mse_loss,
            p2p_mse_loss=p2p_mse_loss,
        )

    def compute_similarity(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        """
        Compute dot-product similarity.

        Args:
          q_reps: [B, H] or [B, 1, H]
          p_reps: [B*G, H] or [B, G, H]

        Returns:
          [B, B*G] or [B, G] depending on input shapes.
        """
        if p_reps.dim() == 2:
            # [B, H] x [B*G, H]^T -> [B, B*G]
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        # [B, 1, H] x [B, G, H]^T -> [B, 1, G] -> broadcast ok
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def _dist_gather_tensor(self, t: Optional[Tensor]) -> Optional[Tensor]:
        """
        All-gather a tensor across processes along the batch dimension.
        """
        if t is None:
            return None
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        # Ensure local rank's tensor is not replaced by a possibly uninitialized buffer
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[dict] = None):
        """
        Enable gradient checkpointing on the encoder.
        """
        if gradient_checkpointing_kwargs is None:
            self.encoder.gradient_checkpointing_enable()
        else:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)