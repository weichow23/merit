# -*- coding: utf-8 -*-
# Author: Yuan Gao
# Date: 2025-09-20
# Project: MERIT
# Description: Experiement code for MERIT

import os
import json
import glob
import random
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from trainer.trainer import get_logger
from qwen_vl_utils import process_vision_info

# --- Global Logger ---
logger = get_logger("train")


# --- Data Preprocessing Utilities ---

def prep_conv(example: Dict, max_img_num: int = 8) -> List[Dict]:
    """
    Formats a raw data example into a structured conversation list suitable for
    Hugging Face's `apply_chat_template`. It handles interleaved text and images.

    Args:
        example (Dict): A dictionary containing conversation data, possibly with an 'image' key.
        max_img_num (int): The maximum number of images to include.

    Returns:
        List[Dict]: A list representing the conversation, typically for a single 'user' turn,
                    followed by an 'assistant' turn if available.
    """
    # Handle multimodal conversations (text + images)
    if "image" in example:
        # Ensure 'image' is a list of paths
        images = [example["image"]] if isinstance(example["image"], str) else example["image"]
        
        # Split the user's text by the <image> placeholder
        text_splits = example["conversations"][0]["value"].split("<image>")
        
        # Sanity check: ensure image placeholders don't exceed the number of provided images
        assert len(text_splits) <= (len(images) + 1), \
            f"Preprocessing Error: Mismatch between <image> tokens and image paths in example: {example}"
        
        # Truncate image list to the maximum allowed number
        images = images[:max_img_num]

        # Interleave text and image content
        content = []
        for text in text_splits:
            if text.strip():
                content.append({"type": "text", "text": text})
            if images:
                # Pop the next image and add it to the content
                content.append({"type": "image", "image": images.pop(0)})
        
        msg = [{"role": "user", "content": content}]
        
    # Handle text-only conversations
    else:
        text = example["conversations"][0]["value"]
        msg = [{"role": "user", "content": [{"type": "text", "text": text}]}]

    # Add the assistant's response if it exists
    if len(example["conversations"]) > 1:
        msg.append({
            "role": "assistant",
            "content": [{"type": "text", "text": example["conversations"][1]["value"]}]
        })
        
    return msg

def inference_left_pad(input_ids: List[torch.Tensor], padding_value: int) -> torch.Tensor:
    """
    Pads a list of 1D tensors to the same length by adding padding to the LEFT.
    Left padding is crucial for correct batch generation in auto-regressive models.

    Args:
        input_ids (List[torch.Tensor]): A list of tensors to pad.
        padding_value (int): The token ID to use for padding.

    Returns:
        torch.Tensor: A 2D tensor of shape (batch_size, max_len) with left padding.
    """
    max_len = max(t.shape[0] for t in input_ids)
    pad_tensors = []
    for tensor in input_ids:
        pad_size = max_len - tensor.shape[0]
        # Prepend padding tokens
        padded_tensor = torch.cat([
            torch.tensor([padding_value] * pad_size, dtype=torch.long),
            tensor
        ])
        pad_tensors.append(padded_tensor)
    return torch.stack(pad_tensors, dim=0)


# --- Dataset Classes ---

class PairwiseCLHardNegDataset(Dataset):
    """
    Dataset for pairwise contrastive learning, where each sample consists of a
    (query, positive_passage, hard_negative_passage) triplet.
    """
    def __init__(self, data_path: str, processor, max_text_length: int = 1024):
        self.max_text_length = max_text_length
        self.processor = processor
        # Expand file path patterns (e.g., 'data/*.jsonl')
        if isinstance(data_path, str):
            data_path = glob.glob(data_path)
        
        # Load data from all specified JSONL files
        self.data = []
        for fp in data_path:
            with open(fp, 'r') as f:
                lines = [json.loads(line) for line in f]
                self.data.extend(lines)
            logger.info(f"Loaded {len(lines)} samples from {fp}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        try:
            record = self.data[idx]
            
            # Prepare conversations for query, positive passage, and a random hard negative
            query_conv = prep_conv(record["query"])
            passage_conv = prep_conv(record["passage"][0])
            hard_neg_index = random.randint(1, len(record["passage"]) - 1)
            passage_neg_conv = prep_conv(record["passage"][hard_neg_index])

            # Apply chat template to get formatted text strings
            query_texts = self.processor.apply_chat_template(query_conv, tokenize=False, add_generation_prompt=True)
            passage_texts = self.processor.apply_chat_template(passage_conv, tokenize=False, add_generation_prompt=True)
            passage_neg_texts = self.processor.apply_chat_template(passage_neg_conv, tokenize=False, add_generation_prompt=True)

            # Process query (text + vision)
            image_inputs, video_inputs = process_vision_info(query_conv)
            query_inputs = self.processor(text=query_texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            # Process positive passage (text + vision)
            image_inputs, video_inputs = process_vision_info(passage_conv)
            passage_inputs = self.processor(text=passage_texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            # Process hard negative passage (text + vision)
            image_inputs, video_inputs = process_vision_info(passage_neg_conv)
            passage_neg_inputs = self.processor(text=passage_neg_texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            return {
                "id": record["id"],
                "query": query_inputs,
                "passage": passage_inputs,
                "passage_neg": passage_neg_inputs,
            }
        
        except Exception as e:
            logger.warning(f"Error processing item at index {idx}: {e}. Skipping and trying next item.")
            return self.__getitem__((idx + 1) % len(self))


class PairwiseCLDataset(Dataset):
    """
    Dataset for pairwise contrastive learning, where each sample consists of a
    (query, positive_passage) pair. Negatives are sourced from other examples in the batch.
    """
    def __init__(self, data_path: str, processor, max_text_length: int = 1024):
        self.max_text_length = max_text_length
        self.processor = processor
        if isinstance(data_path, str):
            data_path = glob.glob(data_path)
            
        self.data = []
        for fp in data_path:
            with open(fp, 'r') as f:
                lines = [json.loads(line) for line in f]
                self.data.extend(lines)
            logger.info(f"Loaded {len(lines)} samples from {fp}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        try:
            record = self.data[idx]
            
            # Prepare conversations for the query and the positive passage
            query_conv = prep_conv(record["query"])
            passage_conv = prep_conv(record["passage"][0])

            # Apply chat template
            query_texts = self.processor.apply_chat_template(query_conv, tokenize=False, add_generation_prompt=True)
            passage_texts = self.processor.apply_chat_template(passage_conv, tokenize=False, add_generation_prompt=True)

            # Process query
            image_inputs, video_inputs = process_vision_info(query_conv)
            query_inputs = self.processor(text=query_texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            # Process passage
            image_inputs, video_inputs = process_vision_info(passage_conv)
            passage_inputs = self.processor(text=passage_texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            return {
                "id": record["id"],
                "query": query_inputs,
                "passage": passage_inputs,
            }
        
        except Exception as e:
            logger.warning(f"Error processing item at index {idx}: {e}. Skipping and trying next item.")
            return self.__getitem__((idx + 1) % len(self))


class InferenceDataset(Dataset):
    """
    Dataset for inference tasks, such as embedding generation. It can filter out
    samples that have already been processed.
    """
    def __init__(self, data_path: str, tgt_folder: str, processor, max_img_num: int = 8, max_text_length: int = 1024,
                 existing_ids: set = None):
        """
        Args:
            existing_ids (set, optional): Precomputed set of already-processed sample IDs
                (as strings), shared identically across all DDP ranks. When running under
                DDP, this MUST be computed once (e.g. on rank 0) and broadcast to every
                rank before this dataset is constructed. If each rank instead scans
                `tgt_folder` independently via `os.path.exists`, faster ranks can finish
                writing .npy files while slower ranks are still starting up, so different
                ranks end up filtering `self.data` to different lengths. DistributedSampler
                partitions indices based on `len(self.data)`, so a mismatch here causes
                samples to be silently skipped. Pass `existing_ids=None` only for
                single-process (non-DDP) runs, where this race condition cannot occur.
        """
        self.max_img_num = max_img_num
        self.max_text_length = max_text_length
        self.processor = processor
        if isinstance(data_path, str):
            data_path = glob.glob(data_path)
            
        # Load all data initially
        self.data = []
        for fp in data_path:
            with open(fp, 'r') as f:
                lines = [json.loads(line) for line in f]
                self.data.extend(lines)
            logger.info(f"Loaded {len(lines)} samples from {fp}")

        logger.info(f"Total samples found: {len(self.data)}")
        
        # Filter out samples for which an output file already exists in the target folder.
        # This is useful for resuming large, interruptible inference jobs.
        if existing_ids is None:
            existing_ids = {
                os.path.splitext(f)[0] for f in os.listdir(tgt_folder) if f.endswith('.npy')
            } if os.path.isdir(tgt_folder) else set()
        self.data = [
            item for item in self.data
            if str(item["id"]) not in existing_ids
        ]
        logger.info(f"Filtered samples to process: {len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        try:
            record = self.data[idx]
            
            # Prepare conversation and get formatted text
            conv = prep_conv(record, self.max_img_num)
            text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            
            # Process text and vision info
            image_inputs, video_inputs = process_vision_info(conv)
            inputs = self.processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            
            return {"id": record["id"], "inputs": inputs}
        
        except Exception as e:
            logger.warning(f"Error processing item at index {idx}: {e}. Skipping and trying next item.")
            return self.__getitem__((idx + 1) % len(self))


# --- Data Collator Classes ---
class DataCollatorForPairwiseCL:
    """
    Collates examples for standard pairwise contrastive learning.
    It combines queries and passages into a single batch for efficient processing.
    """
    def __init__(self, model_max_length: int, pad_token_id: int = 151643):
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id

    def collate_w_padding(self, examples: List[Dict]) -> Dict:
        """Helper function to pad and stack a list of processed examples."""
        input_ids, attention_mask, pixel_values, image_grid_thw = [], [], [], []
        for example in examples:
            input_ids.append(example["input_ids"][0])
            attention_mask.append(example["attention_mask"][0])
            if "pixel_values" in example: pixel_values.append(example["pixel_values"])
            if "image_grid_thw" in example: image_grid_thw.append(example["image_grid_thw"])
        
        # Standard right-padding
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        # Concatenate vision tensors if they exist
        pixel_values = torch.cat(pixel_values, dim=0) if pixel_values else None
        image_grid_thw = torch.cat(image_grid_thw, dim=0) if image_grid_thw else None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def __call__(self, examples: List[Dict]) -> Dict:
        queries = [ex["query"] for ex in examples]
        passages = [ex["passage"] for ex in examples]
        
        # Combine queries and passages to process in one forward pass
        # The first half of the batch will be queries, the second half will be passages.
        all_items = queries + passages
        collated_batch = self.collate_w_padding(all_items)
        return {"passage": collated_batch}


class DataCollatorForPairwiseHardNegCL(DataCollatorForPairwiseCL):
    """
    Collates examples for contrastive learning with hard negatives.
    It combines queries, positive passages, and negative passages into a single batch
    and uses LEFT padding for inference compatibility.
    """
    def collate_w_padding(self, examples: List[Dict]) -> Dict:
        """Overrides base method to use left padding."""
        input_ids, attention_mask, pixel_values, image_grid_thw = [], [], [], []
        for example in examples:
            input_ids.append(example["input_ids"][0])
            attention_mask.append(example["attention_mask"][0])
            if "pixel_values" in example: pixel_values.append(example["pixel_values"])
            if "image_grid_thw" in example: image_grid_thw.append(example["image_grid_thw"])
        
        # Use left padding for autoregressive consistency
        input_ids = inference_left_pad(input_ids, padding_value=self.pad_token_id)
        attention_mask = inference_left_pad(attention_mask, padding_value=0)

        pixel_values = torch.cat(pixel_values, dim=0) if pixel_values else None
        image_grid_thw = torch.cat(image_grid_thw, dim=0) if image_grid_thw else None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def __call__(self, examples: List[Dict]) -> Dict:
        queries = [ex["query"] for ex in examples]
        passages = [ex["passage"] for ex in examples]
        negs = [ex["passage_neg"] for ex in examples]
        
        # Combine all items: batch will be [queries, positives, negatives]
        all_items = queries + passages + negs
        collated_batch = self.collate_w_padding(all_items)
        return {"passage": collated_batch}


class DataCollatorForInference(DataCollatorForPairwiseCL):
    """
    Collates examples for inference. It keeps track of sample IDs and uses
    LEFT padding, which is necessary for batch generation.
    """
    def collate_w_padding(self, examples: List[Dict]) -> Dict:
        """Overrides base method to use left padding."""
        input_ids, attention_mask, pixel_values, image_grid_thw = [], [], [], []
        for example in examples:
            input_ids.append(example["input_ids"][0])
            attention_mask.append(example["attention_mask"][0])
            if "pixel_values" in example: pixel_values.append(example["pixel_values"])
            if "image_grid_thw" in example: image_grid_thw.append(example["image_grid_thw"])

        # Use left padding for autoregressive generation
        input_ids = inference_left_pad(input_ids, padding_value=self.pad_token_id)
        attention_mask = inference_left_pad(attention_mask, padding_value=0)

        pixel_values = torch.cat(pixel_values, dim=0) if pixel_values else None
        image_grid_thw = torch.cat(image_grid_thw, dim=0) if image_grid_thw else None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def __call__(self, examples: List[Dict]) -> Dict:
        id_list = [ex["id"] for ex in examples]
        inputs = [ex["inputs"] for ex in examples]
        
        collated_inputs = self.collate_w_padding(inputs)
        return {"id": id_list, "inputs": collated_inputs}


# --- Example Usage Script ---
if __name__ == "__main__":
    """
    This block serves as a basic test and demonstration of how to use the
    Dataset and DataLoader classes defined above.
    """
    # Define paths and model identifiers (replace with your actual paths)
    MODEL_PATH = "/path/to/your/Qwen2-VL-Instruct"
    DATA_PATH = "/path/to/your/data.jsonl"
    
    # Initialize the processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Instantiate the dataset
    train_dataset = PairwiseCLDataset(data_path=DATA_PATH, processor=processor)
    
    # Instantiate the data collator
    collator = DataCollatorForPairwiseCL(model_max_length=2048, pad_token_id=processor.tokenizer.pad_token_id)
    
    # Create the DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator
    )

    # Fetch and print the first batch to inspect its structure
    print("--- Fetching a sample batch ---")
    for batch in train_loader:
        print("Batch keys:", batch.keys())
        for key, value in batch["passage"].items():
            if value is not None:
                print(f"  - {key}: {value.shape}, {value.dtype}")
            else:
                print(f"  - {key}: None")
        break
    print("--- Test complete ---")