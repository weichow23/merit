# -*- coding: utf-8 -*-
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from data.dataset import InferenceDataset, DataCollatorForInference
from model.modeling_cl import BaseForSCL


def setup(rank: int, world_size: int, ddp_port: str):
    """
    Initializes the distributed process group for a single worker.

    Args:
        rank (int): The unique ID of the current process (worker).
        world_size (int): The total number of processes (workers).
        ddp_port (str): The port number for the master node to coordinate processes.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = ddp_port
    # Initialize the process group with NCCL backend, standard for NVIDIA GPUs
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def inference(rank: int, world_size: int, args: argparse.Namespace):
    """
    The core function executed by each DDP worker to perform inference.

    Args:
        rank (int): The unique ID of the current process.
        world_size (int): The total number of processes.
        args (argparse.Namespace): The command-line arguments.
    """
    # --- 1. DDP and Device Setup ---
    torch.cuda.set_device(rank)
    setup(rank, world_size, args.DDP_port)
    device = torch.device("cuda", rank)

    # --- 2. Model and Processor Loading ---
    # Load the base vision-language model. Using float32 for stable inference.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.float32
    ).to(device)
    
    # Load the processor, which handles tokenization and image preprocessing
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        max_pixels=args.max_img_token * 28 * 28 
    )
    
    # Wrap the base model with the custom SCL head used during training.
    # This provides the `.encode()` method for generating embeddings.
    cl_model = BaseForSCL(encoder_model=model, config=model.config, select_embed="last", inference=True)

    # --- 3. Data Loading ---
    # Create the dataset. It automatically filters out already-processed items.
    infer_set = InferenceDataset(
        data_path=args.data_path,
        tgt_folder=args.tgt_folder,
        processor=processor,
        max_img_num=args.max_img_num
    )
    
    # DistributedSampler ensures each GPU gets a unique, non-overlapping slice of the data.
    sampler = DistributedSampler(
        infer_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    
    # The DataLoader combines the dataset, sampler, and collator to create batches.
    infer_loader = DataLoader(
        infer_set,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=DataCollatorForInference(model_max_length=2048),
        sampler=sampler,
    )

    # --- 4. Inference Loop ---
    # Only show the progress bar on the main worker (rank 0) to avoid clutter.
    prog_bar = tqdm(infer_loader, desc=f"Worker {rank}", disable=(rank != 0))

    # Use torch.inference_mode() for performance (disables gradient tracking).
    with torch.inference_mode():
        for batch in prog_bar:
            pids, inputs = batch["id"], batch["inputs"]
            try:
                # Prepare model inputs, moving all tensors to the correct GPU device.
                model_inputs = {
                    "input_ids": inputs["input_ids"].to(device),
                    "attention_mask": inputs["attention_mask"].to(device),
                    "pixel_values": inputs["pixel_values"].to(device) if inputs["pixel_values"] is not None else None,
                    "image_grid_thw": inputs["image_grid_thw"].to(device) if inputs["image_grid_thw"] is not None else None,
                }
                
                # Generate embeddings using the custom encode method.
                p_reps = cl_model.encode(model_inputs).cpu().numpy().astype(np.float16)
                
                # Save each embedding in the batch to its own file.
                for i, pid in enumerate(pids):
                    # The dataset already filters, but this is a final check.
                    if os.path.exists(os.path.join(args.tgt_folder, str(pid) + '.npy')):
                        continue
                    np.save(os.path.join(args.tgt_folder, str(pid) + '.npy'), p_reps[i])
            except Exception as e:
                # Log errors for specific batches without crashing the whole run.
                print(f"Rank {rank} Error processing batch with IDs {pids}: {e}")
    
    # Clean up the distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Script entry point: parses arguments and spawns DDP processes.
    """
    parser = argparse.ArgumentParser(description="Multi-GPU inference script for generating embeddings.")
    parser.add_argument("--model_name", "-m", type=str, required=True, help="Path or HF identifier for the trained model checkpoint.")
    parser.add_argument("--data_path", "-d", type=str, required=True, help="Path to the input data file or glob pattern (e.g., 'data/*.jsonl').")
    parser.add_argument("--tgt_folder", "-t", type=str, required=True, help="Directory to save the output .npy embedding files.")
    parser.add_argument("--max_img_num", type=int, default=8, help="Maximum number of images to process per sample.")
    parser.add_argument("--max_img_token", type=int, default=576, help="Maximum number of images to process per sample.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU.")
    parser.add_argument("--world_size", type=int, default=8, help="Total number of GPUs (processes) to use.")
    parser.add_argument("--DDP_port", type=str, default="10086", help="Port for DDP communication.")
    args = parser.parse_args()

    print("--- Inference Configuration ---")
    print(args)
    print("-----------------------------")

    # Ensure the target directory for embeddings exists.
    os.makedirs(args.tgt_folder, exist_ok=True)
    
    # Use mp.spawn to launch `world_size` instances of the `inference` function.
    # Each instance will be a separate process with its own rank.
    mp.spawn(inference, args=(args.world_size, args), nprocs=args.world_size, join=True)