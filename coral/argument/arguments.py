# -*- coding: utf-8 -*-
# Author: Yuan Gao
# Date: 2025-09-20
# Project: MERIT
# Description: Experiement code for MERIT

from dataclasses import dataclass, field
from typing import Optional
import transformers

# --- Training Configuration ---
@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    """
    Arguments related to the training process, extending Hugging Face's Seq2SeqTrainingArguments
    with custom options for project management, model freezing, and LoRA.
    """
    # -- Project Management --
    project_name: str = field(
        default=None,
        metadata={"help": "The name of the project, used for logging and tracking (e.g., with Weights & Biases)."}
    )
    
    # -- Model Freezing --
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "If True, freezes the weights of the vision encoder part of the model."}
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "If True, freezes the weights of the language model backbone."}
    )
    
    # -- LoRA (Low-Rank Adaptation) Configuration --
    lora_enable: bool = field(
        default=False,
        metadata={"help": "If True, enables LoRA for efficient fine-tuning."}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "The rank 'r' of the LoRA update matrices."}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha scaling factor for LoRA (scales the learned weights)."}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Specifies which bias parameters to train. Options: 'none', 'all', 'lora_only'."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "The dropout probability for the LoRA layers."}
    )
    
    # -- Precision Control --
    fp32: bool = field(
        default=False,
        metadata={"help": "If True, forces training in full 32-bit floating point precision, disabling mixed precision (fp16/bf16)."}
    )

# --- Data Configuration ---

@dataclass
class DataArguments:
    """
    Arguments related to data loading and processing.
    """
    data_path: str = field(
        default=None,
        metadata={"help": "The file path to the training dataset."}
    )

# --- Model Architecture Configuration ---

@dataclass
class ModelArguments:
    """
    Arguments related to the model architecture and its core configuration.
    """
    model_base: Optional[str] = field(
        default=None,
        metadata={"help": "The base pre-trained model or path to be used as a backbone (e.g., a LLaMA variant)."}
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model checkpoint to load. Can be a local path or a Hugging Face Hub identifier."}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "The type of model architecture to use (e.g., 'llava', 'qwen-vl'). Helps in loading custom code."}
    )
    model_max_length: int = field(
        default=4096,
        metadata={"help": "The maximum sequence length (in tokens) for the model's input."}
    )
    max_image_token: int = field(
        default=1280,
        metadata={"help": "The maximum number of visual tokens for image inputs after resizing."}
    )

# --- Supervised Contrastive Learning (SCL) Configuration ---

@dataclass
class SCLArguments:
    """
    Arguments for configuring a Supervised Contrastive Learning (SCL) objective.
    """
    contrastive_loss: str = field(
        default="InfoNCE",
        metadata={"help": "The type of contrastive loss function to use."}
    )
    contrastive_temp: float = field(
        default=1.0,
        metadata={"help": "The temperature hyperparameter for the contrastive loss."}
    )
    negatives_cross_device: bool = field(
        default=False,
        metadata={"help": "If True, gathers negative samples from all devices in a distributed setting."}
    )
    use_inbatch_neg: bool = field(
        default=True,
        metadata={"help": "If True, uses other samples within the same batch as negative examples."}
    )
    select_embed: str = field(
        default="last",
        metadata={"help": "Which embedding to select for the contrastive loss (e.g., 'last' for the last hidden state)."}
    )
    has_hardneg: bool = field(
        default=False,
        metadata={"help": "Specifies whether the dataset provides explicit hard negative examples."}
    )

# --- CORAL Training Strategy Configuration ---

@dataclass
class CORALArguments:
    """
    Arguments for a specific training strategy, possibly related to the 'CORAL' method,
    involving a separate decoder and masked modeling objectives.
    """
    pretrained_decoder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pre-trained decoder to be loaded separately."}
    )
    decoder_lr: float = field(
        default=1e-5,
        metadata={"help": "The learning rate specifically for the decoder part of the model."}
    )
    l_mask_prob: float = field(
        default=0.5,
        metadata={"help": "The probability of masking tokens in the language portion of the input."}
    )
    v_mask_prob: float = field(
        default=0.5,
        metadata={"help": "The probability of masking patches in the vision portion of the input."}
    )
    lambda_l: float = field(
        default=0.5,
        metadata={"help": "The weight (lambda) for the language reconstruction loss."}
    )
    lambda_v: float = field(
        default=0.5,
        metadata={"help": "The weight (lambda) for the vision reconstruction loss."}
    )