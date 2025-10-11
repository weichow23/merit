# -*- coding: utf-8 -*-
# Author: Yuan Gao
# Date: 2025-09-20
# Project: MERIT
# Description: Experiement code for MERIT

import argparse
import os
import random
import pathlib
import warnings

import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import AutoProcessor, Trainer, set_seed
from peft import LoraConfig, get_peft_model
from dataclasses import asdict

# --- Local Module Imports ---
from data.dataset import PairwiseCLDataset, DataCollatorForPairwiseCL, DataCollatorForPairwiseHardNegCL, PairwiseCLHardNegDataset
from argument.arguments import ModelArguments, DataArguments, TrainingArguments, SCLArguments, CORALArguments
from trainer.trainer import EmbeddingCallback, EmbeddingTrainer, print_trainable_params, find_all_linear_names, get_logger
from model.modeling_cl import BaseForSCL
from model.modeling_coral import BaseForCORAL
from model.modeling_base import (
    Qwen2VLForConditionalGenerationWithEmbedding,
    Qwen2_5_VLForConditionalGenerationWithEmbedding,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)

# --- Initial Setup ---
warnings.filterwarnings("ignore")
os.environ["NCCL_DEBUG"] = ""  # Set NCCL debug level, empty means default
logger = get_logger("train")


# --- Model Configuration Registries ---

# Maps a model type string to its corresponding wrapper class, argument class name,
# dataset class, and data collator class. This acts as a dispatcher.
model_cls_dict = {
    "scl": (BaseForSCL, "scl_args", PairwiseCLDataset, DataCollatorForPairwiseCL),
    "sclwneg": (BaseForSCL, "scl_args", PairwiseCLHardNegDataset, DataCollatorForPairwiseHardNegCL),
    "coral": (BaseForCORAL, "coral_args", PairwiseCLDataset, DataCollatorForPairwiseCL),
    "coralwneg": (BaseForCORAL, "coral_args", PairwiseCLHardNegDataset, DataCollatorForPairwiseHardNegCL),
}

# Maps a base model name to its standard Hugging Face class and our custom
# embedding-enabled variant.
base_model_dict = {
    "qwen2vl": (Qwen2VLForConditionalGeneration, Qwen2VLForConditionalGenerationWithEmbedding),
    "qwen2.5vl": (Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLForConditionalGenerationWithEmbedding)
}


def setup_seed(seed: int):
    """
    Sets the random seed for reproducibility across all relevant libraries.
    
    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set cudnn to be deterministic, which can slightly slow down training
    # but ensures reproducibility.
    try:
        torch.backends.cudnn.deterministic = True
    except Exception:
        logger.warning("Could not set `torch.backends.cudnn.deterministic` to True.")


def train(args: argparse.Namespace):
    """
    The main function to set up and run the training process.

    Args:
        args (argparse.Namespace): Command-line arguments, expected to contain `conf` path.
    """
    # --- 1. Argument Parsing ---
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, SCLArguments, CORALArguments))
    model_args, data_args, training_args, scl_args, coral_args = parser.parse_yaml_file(args.conf)

    # --- 2. Environment and Initial Setup ---
    os.environ["WANDB_PROJECT"] = training_args.project_name
    training_args.learning_rate = float(training_args.learning_rate)

    # Sanity checks for model configuration
    assert model_args.model_type in model_cls_dict, f"Unsupported Model Type: {model_args.model_type}"
    assert model_args.model_base in base_model_dict, f"Unsupported Base Model: {model_args.model_base}"

    # --- 3. Model Loading and Configuration ---
    base_model_cls, base_model_mae_cls = base_model_dict[model_args.model_base]
    
    # Load the appropriate base model. 'mae' types (like CORAL) may have a different architecture.
    if model_args.model_type in ["coral", "coralwneg"]:
        model = base_model_mae_cls.from_pretrained(model_args.model_name, torch_dtype=torch.bfloat16)
        # Use a separate learning rate for the decoder if specified
        training_args.decoder_lr = coral_args.decoder_lr
    else:
        model = base_model_cls.from_pretrained(model_args.model_name, torch_dtype=torch.bfloat16)
        # Use the main learning rate for all parts of the model
        training_args.decoder_lr = training_args.learning_rate

    model.config.use_cache = False  # Disable caching for training
    
    # Set model precision
    if training_args.bf16:
        model.to(torch.bfloat16)
    elif training_args.fp16:
        model.to(torch.float16)

    # --- 4. LoRA (Low-Rank Adaptation) Setup ---
    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model, training_args.freeze_vision_tower),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if dist.get_rank() == 0:
            logger.info("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # --- 5. Gradient Checkpointing ---
    if training_args.gradient_checkpointing:
        # A memory-saving technique that re-computes activations during the backward pass.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # --- 6. Processor, Dataset, and Model Wrapping ---
    processor = AutoProcessor.from_pretrained(model_args.model_name, trust_remote_code=True, max_pixels=model_args.max_image_token * 28 * 28) 
    
    # Select the correct model wrapper, arguments, dataset, and collator based on model_type
    model_cls, args_cls_name, data_cls, collator_cls = model_cls_dict[model_args.model_type]
    
    # Dynamically select the correct argument object (scl_args or coral_args)
    task_args = locals()[args_cls_name]
    
    # Instantiate the model wrapper (e.g., BaseForSCL) with the base model and task-specific arguments
    if model_args.model_type not in ["coral", "coralwneg"]:
        model_ = model_cls(encoder_model=model, config=model.config, **asdict(task_args))
    else:
        model_ = model_cls(encoder_model=model, config=model.config, **asdict(coral_args), **asdict(scl_args))
    
    # Initialize the training dataset
    trainset = data_cls(data_path=data_args.data_path, processor=processor)

    # --- 7. Model Freezing & Parameter Setup ---
    
    # Load pretrained decoder weights if provided (for CORAL-like models)
    if "mae" in model_args.model_type and getattr(coral_args, "pretrained_decoder", None):
        if dist.get_rank() == 0:
            logger.info(f"Loading decoder weights from {coral_args.pretrained_decoder}")
        model_.q2p_decoder.load_state_dict(torch.load(os.path.join(coral_args.pretrained_decoder, "q2p_decoder.bin")))
        model_.p2p_decoder.load_state_dict(torch.load(os.path.join(coral_args.pretrained_decoder, "p2p_decoder.bin")))
        # Ensure these loaded parts are trainable
        for name, param in model_.named_parameters():
            if "_decoder" in name or "vision_head" in name or "decoder_position_embeddings" in name:
                param.requires_grad = True

    # Freeze the vision tower weights
    if training_args.freeze_vision_tower:
        if dist.get_rank() == 0:
            logger.info("Freezing vision tower...")
        for name, param in model_.named_parameters():
            if "visual" in name:
                param.requires_grad = False
    
    # Freeze the main language model backbone
    if training_args.freeze_backbone:
        if dist.get_rank() == 0:
            logger.info("Freezing language model backbone...")
        for name, param in model_.named_parameters():
            if "encoder.model" in name:
                param.requires_grad = False
    
    # If LoRA is enabled, freeze everything except LoRA and specific tunable modules
    if training_args.lora_enable:
        if dist.get_rank() == 0:
            logger.info("Setting trainable parameters for LoRA...")
        for name, param in model_.named_parameters():
            if "lora" in name or "embed_tokens" in name or "_decoder" in name or "vision_head" in name or "decoder_position_embeddings" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Print a summary of trainable parameters for verification
    print_trainable_params(model_)

    # --- 8. Trainer Initialization ---
    trainer = EmbeddingTrainer(
        model=model_,
        args=training_args,
        train_dataset=trainset,
        data_collator=collator_cls(model_max_length=model_args.model_max_length),
        callbacks=[EmbeddingCallback],
    )

    # --- 9. Start Training ---
    # Check if there are existing checkpoints in the output directory to resume from
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        if dist.get_rank() == 0:
            logger.info("Resuming training from the latest checkpoint.")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # --- 10. Saving the Final Model ---
    trainer.save_state()
    trainer.save_model() # Saves the full training wrapper (e.g., BaseForSCL)

    # Create a dedicated directory for the final encoder model
    encoder_output_dir = os.path.join(training_args.output_dir, "encoder")
    os.makedirs(encoder_output_dir, exist_ok=True)

    # Save the encoder part of the model. Logic differs for LoRA.
    if training_args.lora_enable:
        if trainer.is_world_process_zero():
            logger.info("Merging LoRA adapters and saving the final encoder model.")
            # Merge LoRA weights into the base model for easy inference
            encoder_model = model_.encoder.merge_and_unload()
            encoder_model.config.save_pretrained(encoder_output_dir)
            encoder_model.save_pretrained(encoder_output_dir)
    else:
        # For full fine-tuning, just save the encoder directly
        encoder_model = model_.encoder
        encoder_model.save_pretrained(encoder_output_dir)

    # Save decoder weights separately if they were trained (e.g., CORAL)
    if training_args.freeze_backbone:
        decoder_output_dir = os.path.join(training_args.output_dir, "decoder")
        os.makedirs(decoder_output_dir, exist_ok=True)
        torch.save(model_.q2p_decoder.state_dict(), os.path.join(decoder_output_dir, "q2p_decoder.bin"))
        torch.save(model_.p2p_decoder.state_dict(), os.path.join(decoder_output_dir, "p2p_decoder.bin"))


if __name__ == "__main__":
    """
    Script entry point. Parses command-line arguments and launches the training process.
    """
    # Set a fixed seed for reproducibility
    setup_seed(seed=42)
    
    # Define command-line argument parser
    parser = argparse.ArgumentParser(description="Run model training from a configuration file.")
    parser.add_argument(
        "--conf", type=str, required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    
    # Start the training
    train(args)