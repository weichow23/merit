# -*- coding: utf-8 -*-
# Author: Yuan Gao, Linfeng Li, Xian Wang
# Date: 2025-09-20
# Project: MERIT
# Description: Experiement code for MERIT

from typing import Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.file_utils import ModelOutput
from dataclasses import dataclass


@dataclass
class EncoderOutput(ModelOutput):
    """
    Output container for SCL forward pass.
    """
    q_reps: Optional[Tensor] = None   # [B, H] query representations
    p_reps: Optional[Tensor] = None   # [B*G, H] passage (and possibly hard-negative) representations
    loss: Optional[Tensor] = None     # scalar contrastive loss (train mode); None in eval
    scores: Optional[Tensor] = None   # similarity matrix; [B, B*G] in train/eval


class BaseForSCL(nn.Module):
    """
    Simple Contrastive Learning (SCL) base:
      - Single encoder produces representations for concatenated inputs.
      - Supports optional hard negatives: inputs shaped as (q, p[, h]).
      - Contrastive loss can use in-batch and/or cross-device negatives.
    """

    def __init__(self, encoder_model: nn.Module, config, **kwargs):
        super().__init__()
        self.config = config
        self.encoder = encoder_model

        # How to pool token embeddings into a single vector per sequence
        # Supported: "mean", "last" (CLS-like by last token). "head" is not implemented.
        self.select_embed: str = kwargs["select_embed"]

        # Inference mode disables training-specific knobs
        self.inference: bool = kwargs.get("inference", False)

        if not self.inference:
            self.contrastive_temp: float = kwargs["contrastive_temp"]
            self.negatives_cross_device: bool = kwargs["negatives_cross_device"]
            self.use_inbatch_neg: bool = kwargs["use_inbatch_neg"]
            self.has_hardneg: bool = kwargs.get("has_hardneg", False)

            # Use default reduction='mean' (avoid deprecated reduce="mean")
            self.loss_fn = nn.CrossEntropyLoss()

            # Cross-device negatives require initialized torch.distributed
            if self.negatives_cross_device:
                if not dist.is_initialized():
                    raise ValueError(
                        "Distributed training not initialized for cross-device negatives."
                    )
                self.process_rank = dist.get_rank()
                self.world_size = dist.get_world_size()
        else:
            # For inference, assume no contrastive loss; hardneg flag may still control splitting.
            self.contrastive_temp = 0.0
            self.negatives_cross_device = False
            self.use_inbatch_neg = False
            self.has_hardneg = kwargs.get("has_hardneg", False)

    def get_model(self) -> nn.Module:
        """Expose the underlying encoder."""
        return self.encoder

    def encode(self, features: Optional[Dict[str, Tensor]]) -> Optional[Tensor]:
        """
        Encode inputs and return normalized sequence representations.

        Args:
          features: Dict containing tokenized inputs for the encoder.

        Returns:
          reps: [B, H] L2-normalized representations, or None if features is None.
        """
        if features is None:
            return None

        outputs = self.encoder(**features, return_dict=True, output_hidden_states=True)
        last_hidden_states: Tensor = outputs.hidden_states[-1]  # [B, L, H]

        # Pooling strategy
        if self.select_embed == "mean":
            reps = last_hidden_states.mean(dim=1)  # [B, H]
        elif self.select_embed == "last":
            reps = last_hidden_states[:, -1, :]    # [B, H]
        elif self.select_embed == "head":
            # Placeholder for potential CLS/head token pooling if present
            raise NotImplementedError("select_embed='head' is not implemented.")
        else:
            raise ValueError(f"Unknown select_embed='{self.select_embed}'")

        # Normalize to unit length for cosine-sim via dot-product
        reps = F.normalize(reps, p=2, dim=-1)
        return reps.contiguous()

    def forward(
        self,
        query: Optional[Dict[str, Tensor]] = None,
        passage: Optional[Dict[str, Tensor]] = None,
        passage_neg: Optional[Dict[str, Tensor]] = None,
    ) -> Union[Tuple, EncoderOutput]:
        """
        Forward pass:
          - Encode concatenated inputs in 'passage' as reps = [q; p; (h)].
          - Split into q/p(/h) according to has_hardneg.
          - If training:
              * Optionally gather cross-device negatives.
              * Compute contrastive logits and CE loss.
          - Else (eval):
              * Return similarity scores without loss.

        Args:
          query: Unused here (kept for API compatibility).
          passage: Dict containing concatenated inputs (q, p[, h]).
          passage_neg: Unused when negatives are concatenated in 'passage'.

        Returns:
          EncoderOutput with loss (train), scores, and split reps.
        """
        reps = self.encode(passage)  # [B_total, H] with B_total = 2B or 3B
        if reps is None:
            return EncoderOutput()

        # Ensure normalized (encode already normalizes, but keep defensively)
        reps = F.normalize(reps, p=2, dim=-1)

        # Split into q/p(/h)
        if self.has_hardneg:
            third = len(reps) // 3
            q_reps = reps[:third]
            p_reps = reps[third : 2 * third]
            h_reps = reps[2 * third :]
        else:
            half = len(reps) // 2
            q_reps = reps[:half]
            p_reps = reps[half:]
            h_reps = None

        # Training vs. evaluation branches
        if self.training:
            # Optionally gather representations across devices to enlarge negative pool
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                if self.has_hardneg and h_reps is not None:
                    h_reps = self._dist_gather_tensor(h_reps)
                    # Concatenate hard negatives after positives for each query
                    p_reps = torch.cat([p_reps, h_reps], dim=0)

            # Number of passages (and hard negatives) per query
            group_size = p_reps.size(0) // q_reps.size(0)
            temp = max(self.contrastive_temp, 1e-8)  # avoid divide-by-zero

            if self.use_inbatch_neg:
                # Full in-batch cross-similarity: [B, B*G]
                scores = self.compute_similarity(q_reps, p_reps) / temp
                scores = scores.view(q_reps.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.loss_fn(scores, target)
            else:
                # Only positives grouped per query: p_reps -> [B, G, H]
                # Compare q_reps [B, 1, H] against its G passages [B, G, H]
                scores = (
                    self.compute_similarity(
                        q_reps[:, None, :], p_reps.view(q_reps.size(0), group_size, -1)
                    ).squeeze(1)
                    / temp
                )  # [B, G]
                scores = scores.view(q_reps.size(0), -1)
                # Positive is index 0 assuming layout [pos, (other pos/hardnegs...)]
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.loss_fn(scores, target)
        else:
            # Evaluation mode: return similarity matrix for retrieval
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_similarity(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        """
        Compute dot-product similarity.

        Args:
          q_reps: [B, H] or [B, 1, H]
          p_reps: [B*G, H] or [B, G, H]

        Returns:
          [B, B*G] or [B, G] similarity tensor.
        """
        if p_reps.dim() == 2:
            # [B, H] x [B*G, H]^T -> [B, B*G]
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        # [B, 1, H] x [B, G, H]^T -> [B, 1, G]
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def _dist_gather_tensor(self, t: Optional[Tensor]) -> Optional[Tensor]:
        """
        All-gather a tensor across processes along the batch dimension.

        Note: Assumes all ranks have the same local batch size.
        """
        if t is None:
            return None
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        # Replace with the local tensor to avoid any async issues
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