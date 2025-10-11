# -*- coding: utf-8 -*-
# Author: Yuan Gao
# Date: 2025-09-20
# Project: MERIT
# Description: Experiement code for MERIT

import os
import re
import sys
import logging
from typing import Any, Dict, Union, List, Set

import torch
import torch.distributed as dist
from prettytable import PrettyTable
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl

# --- Utilities ---

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a standard console logger.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers if they already exist (e.g., in interactive sessions)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


def print_trainable_params(model: torch.nn.Module):
    """
    Prints a summary of trainable parameters in a neat table, grouped by layer.

    Args:
        model (torch.nn.Module): The model to inspect.
    """
    if dist.get_rank() == 0:
        trainable_params = [k for k,v in model.named_parameters() if v.requires_grad]
        trainable_params_group = {}
        for para in trainable_params:
            layer_num = re.findall(r'layers.(\d+)\.',para)
            block_num = re.findall(r'blocks.(\d+)\.',para)
            if layer_num:
                cur_layer = int(layer_num[0])
                if para.replace('layers.'+layer_num[0],'layers.*') not in trainable_params_group:
                    trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')] = layer_num[0]
                elif cur_layer > int(trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')]):
                    trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')] = layer_num[0]
            elif block_num:
                cur_layer = int(block_num[0])
                if para.replace('blocks.'+block_num[0],'blocks.*') not in trainable_params_group:
                    trainable_params_group[para.replace('blocks.'+block_num[0],'blocks.*')] = block_num[0]
                elif cur_layer > int(trainable_params_group[para.replace('blocks.'+block_num[0],'blocks.*')]):
                    trainable_params_group[para.replace('blocks.'+block_num[0],'blocks.*')] = block_num[0]
            else:
                trainable_params_group[para] = '0'
        table = PrettyTable(['Parameter Name','Max Layer Number'])
        for key in trainable_params_group.keys():
            table.add_row([key, str(int(trainable_params_group[key])+1)])
        
        print(table)
        total_num = sum([v.numel() for k,v in model.named_parameters()])
        trainable_num = sum([v.numel() for k,v in model.named_parameters() if v.requires_grad])
        print('Total: {:.2f}M'.format(total_num/1e6))
        print('Trainable: {:.2f}M'.format(trainable_num/1e6))


def find_all_linear_names(model: torch.nn.Module, freeze_vision_tower: bool) -> List[str]:
    """
    Finds all linear layer names suitable for LoRA adaptation.

    Args:
        model (torch.nn.Module): The model to inspect.
        freeze_vision_tower (bool): If True, excludes vision-related modules.

    Returns:
        List[str]: A list of module names to target with LoRA.
    """
    cls = torch.nn.Linear
    lora_module_names: Set[str] = set()

    # Keywords to identify vision-related components
    multimodal_keywords = ["visual", "vision_model"] if freeze_vision_tower else []

    for name, module in model.named_modules():
        # Skip vision modules if they are frozen
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        
        if isinstance(module, cls):
            names = name.split(".")
            # Get the last part of the name (e.g., "q_proj" from "model.layers.0.attn.q_proj")
            lora_module_names.add(names[-1])

    # Commonly excluded modules for stability or efficiency
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    if "embed_tokens" in lora_module_names:
        lora_module_names.remove("embed_tokens")

    return list(lora_module_names)


def get_optimizer_groups(model: torch.nn.Module, training_args: TrainingArguments) -> List[Dict[str, Any]]:
    """
    Creates optimizer parameter groups with different learning rates and weight decay settings.
    Specifically, it separates decoder parameters from the rest of the model.

    Args:
        model (torch.nn.Module): The model to configure.
        training_args (TrainingArguments): Arguments containing learning rates and weight decay.

    Returns:
        List[Dict]: A list of parameter groups for the optimizer.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    
    # Identifiers for decoder modules (e.g., in CORAL/MAE models)
    decoder_keywords = ("q2p_decoder", "p2q_decoder")

    optimizer_grouped_parameters = []

    # --- 1. Decoder Parameters (With Weight Decay) ---
    optimizer_grouped_parameters.append({
        "params": [
            p for n, p in model.named_parameters()
            if any(k in n for k in decoder_keywords)
            and not any(nd in n for nd in no_decay)
        ],
        "lr": training_args.decoder_lr,
        "weight_decay": training_args.weight_decay,
    })

    # --- 2. Decoder Parameters (Without Weight Decay) ---
    optimizer_grouped_parameters.append({
        "params": [
            p for n, p in model.named_parameters()
            if any(k in n for k in decoder_keywords)
            and any(nd in n for nd in no_decay)
        ],
        "lr": training_args.decoder_lr,
        "weight_decay": 0.0,
    })

    # --- 3. Other Parameters (With Weight Decay) ---
    optimizer_grouped_parameters.append({
        "params": [
            p for n, p in model.named_parameters()
            if not any(k in n for k in decoder_keywords)
            and not any(nd in n for nd in no_decay)
        ],
        "lr": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
    })

    # --- 4. Other Parameters (Without Weight Decay) ---
    optimizer_grouped_parameters.append({
        "params": [
            p for n, p in model.named_parameters()
            if not any(k in n for k in decoder_keywords)
            and any(nd in n for nd in no_decay)
        ],
        "lr": training_args.learning_rate,
        "weight_decay": 0.0,
    })

    # Filter out any groups that are empty
    return [group for group in optimizer_grouped_parameters if group["params"]]


# --- Callbacks ---

class EmbeddingCallback(TrainerCallback):
    """
    A custom callback to save the encoder and decoder models at each checkpoint.
    This is crucial for models where the trained entity is a wrapper around the base model.
    """
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Save at the specified checkpointing interval
        if state.global_step % args.save_steps == 0 and state.global_step > 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            encoder_folder = os.path.join(checkpoint_dir, "encoder")
            os.makedirs(encoder_folder, exist_ok=True)

            model_wrapper = kwargs["model"]

            # --- Save Encoder ---
            if getattr(args, "lora_enable", False):
                # If LoRA is used, merge the adapters before saving for easy inference.
                # We assume the wrapper has an 'encoder' attribute with a 'merge_and_unload' method.
                try:
                    encoder_model = model_wrapper.encoder.merge_and_unload()
                    encoder_model.save_pretrained(encoder_folder)
                except Exception as e:
                    get_logger("train").warning(f"Could not merge and save LoRA encoder at step {state.global_step}: {e}")
            else:
                # Save the encoder directly
                model_wrapper.encoder.save_pretrained(encoder_folder)

            # --- Save Decoder (if applicable) ---
            # Save decoder components if the backbone was frozen (implying decoder training).
            if getattr(args, "freeze_backbone", False) and hasattr(model_wrapper, "q2p_decoder"):
                decoder_folder = os.path.join(checkpoint_dir, "decoder")
                os.makedirs(decoder_folder, exist_ok=True)

                # Save decoder state_dicts
                torch.save(model_wrapper.q2p_decoder.state_dict(), os.path.join(decoder_folder, "q2p_decoder.bin"))
                if hasattr(model_wrapper, "p2p_decoder"):
                     torch.save(model_wrapper.p2p_decoder.state_dict(), os.path.join(decoder_folder, "p2p_decoder.bin"))

        return control


# --- Trainer Extension ---

class EmbeddingTrainer(Trainer):
    """
    Custom Trainer subclass that overrides logging and the training step
    to handle complex, multi-part loss functions (e.g., for CORAL or SCL).
    """

    def log(self, logs: Dict[str, float], eval = False) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        
        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


    def training_step(
        self, model: torch.nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        del inputs

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        loss = loss / self.args.gradient_accumulation_steps

        if (self.state.global_step % self.args.logging_steps == 0)  and (not outputs.get("decoder_loss", None) is None) and (outputs.get("q2p_loss", None) is None):
            encoder_loss = outputs['encoder_loss']
            decoder_loss = outputs['decoder_loss']
            mlm_loss = outputs["mlm_loss"]
            mse_loss = outputs["mse_loss"]
            
            if self.args.n_gpu > 1:
                encoder_loss = encoder_loss.mean()
                decoder_loss = decoder_loss.mean()
                mlm_loss = mlm_loss.mean()
                mse_loss = mse_loss.mean()

            self.log({"encoder_loss": encoder_loss.detach().item(),"decoder_loss":decoder_loss.detach().item(), "mlm_loss": mlm_loss.detach().item(), "mse_loss": mse_loss.detach().item()})
        
        elif (self.state.global_step % self.args.logging_steps == 0)  and (not outputs.get("contrastive_loss", None) is None):
            contrastive_loss = outputs['contrastive_loss']
            mae_loss = outputs['mae_loss']
            mlm_loss = outputs["mlm_loss"]
            mse_loss = outputs["mse_loss"]
            q2p_loss = outputs["q2p_loss"]
            p2p_loss = outputs["p2p_loss"]
            q2p_mlm_loss = outputs["q2p_mlm_loss"]
            p2p_mlm_loss = outputs["p2p_mlm_loss"]
            q2p_mse_loss = outputs["q2p_mse_loss"]
            p2p_mse_loss = outputs["p2p_mse_loss"]
            
            if self.args.n_gpu > 1:
                contrastive_loss = contrastive_loss.mean()
                decoder_loss = decoder_loss.mean()
                mlm_loss = mlm_loss.mean()
                mse_loss = mse_loss.mean()
                q2p_loss = q2p_loss.mean()
                p2p_loss = p2p_loss.mean()
                q2p_mlm_loss = q2p_mlm_loss.mean()
                p2p_mlm_loss = p2p_mlm_loss.mean()
                q2p_mse_loss = q2p_mse_loss.mean()
                p2p_mse_loss = p2p_mse_loss.mean()

            self.log({
                    "contrastive_loss": contrastive_loss.detach().item(),
                    "mae_loss": mae_loss.detach().item(),
                    "mlm_loss": mlm_loss.detach().item(),
                    "mse_loss": mse_loss.detach().item(),
                    "q2p/q2p_loss": q2p_loss.detach().item(),
                    "q2p/q2p_mlm_loss": q2p_mlm_loss.detach().item(),
                    "q2p/q2p_mse_loss": q2p_mse_loss.detach().item(),
                    "p2p/p2p_loss": p2p_loss.detach().item(),
                    "p2p/p2p_mlm_loss": p2p_mlm_loss.detach().item(),
                    "p2p/p2p_mse_loss": p2p_mse_loss.detach().item()
                })        
            
        self.accelerator.backward(loss)

        return loss.detach()