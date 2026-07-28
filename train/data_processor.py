import io
import json
import os
import random
from typing import Any

import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import HunYuanVLProcessor


class VLDataset(Dataset):
    """Custom dataset for vision-language data."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        processor: HunYuanVLProcessor,
        image_lmdb_path: str | None = None,
        image_lmdb_root: str | None = None,
        max_length: int = 2048,
        is_packed: bool = False,
    ):
        super().__init__()
        self.processor = processor
        self.max_length = max_length
        self.image_folder = image_folder
        self.is_packed = is_packed

        # Load data from one or more files (comma-separated paths supported)
        # Supports both JSON (.json) and JSONL (.jsonl) formats
        data_paths = [p.strip() for p in data_path.split(",") if p.strip()]
        raw_data = []
        for dp in data_paths:
            if dp.endswith(".jsonl"):
                # JSONL format: each line is a JSON object (packed: each line is a JSON array)
                with open(dp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            raw_data.append(json.loads(line))
                print(f"Loaded JSONL file: {dp}")
            else:
                # JSON format: entire file is a JSON array
                with open(dp, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                if isinstance(file_data, list):
                    raw_data.extend(file_data)
                else:
                    raw_data.append(file_data)
                print(f"Loaded JSON file: {dp}")

        # Handle packed data format: [[item1, item2, ...], [item3, item4, ...], ...]
        # Each inner list is a pre-packed batch that should be processed together
        if is_packed and len(raw_data) > 0 and isinstance(raw_data[0], list):
            # Keep the packed structure
            self.data = raw_data
            total_items = sum(len(pack) for pack in raw_data)
            print(
                f"Loaded packed dataset: {len(raw_data)} packs, {total_items} total items from {len(data_paths)} file(s)"
            )
        else:
            # Normal format: [item1, item2, ...]
            self.data = raw_data
            print(f"Loaded dataset: {len(self.data)} items from {len(data_paths)} file(s)")

        # Image storage configuration. Three modes:
        #   1) image_lmdb_root != None  -> per-source LMDBs at <root>/<source>
        #      sample dict must carry "source" + "image_id" (int).
        #   2) image_lmdb_path != None  -> single legacy LMDB; sample dict
        #      must carry "image_id" (int).
        #   3) neither                  -> read images from disk under
        #      image_folder using sample["image"] path.
        # NOTE: LMDB envs do NOT survive a DataLoader worker fork. We DO NOT
        # open them in __init__; instead the first __getitem__ inside each
        # worker opens its own envs lazily (cached on the dataset instance).
        self.image_lmdb_root = image_lmdb_root
        self.image_lmdb_path = image_lmdb_path
        self._lmdb_envs = None  # lazy: dict[source_name -> (env, txn)] OR ("__single__", env, txn)

    def _get_lmdb_txn(self, source: str | None = None):
        """Lazily open LMDB env(s) inside the current worker process.

        Returns the txn for ``source`` (multi-LMDB mode) or the single txn
        (legacy single-LMDB mode), or None if neither is configured.
        """
        if self.image_lmdb_root is None and self.image_lmdb_path is None:
            return None

        if self._lmdb_envs is None:
            self._lmdb_envs = {}

        if self.image_lmdb_root is not None:
            if source is None:
                raise ValueError("image_lmdb_root is set but sample is missing 'source' field")
            if source not in self._lmdb_envs:
                lmdb_dir = os.path.join(self.image_lmdb_root, source)
                env = lmdb.open(
                    lmdb_dir,
                    max_readers=128,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                self._lmdb_envs[source] = (env, env.begin(buffers=False))
            return self._lmdb_envs[source][1]
        else:
            # legacy single LMDB
            if "__single__" not in self._lmdb_envs:
                env = lmdb.open(
                    self.image_lmdb_path,
                    max_readers=128,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                self._lmdb_envs["__single__"] = (env, env.begin(buffers=False))
            return self._lmdb_envs["__single__"][1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # If packed, return a list of items (a pre-packed batch)
        # If not packed, return a single item
        if self.is_packed and isinstance(self.data[idx], list):
            # data[idx] is a list of items that should be processed together
            pack = self.data[idx]
            results = []
            for item in pack:
                try:
                    results.append(self._process_single_item(item))
                except Exception as e:
                    print(f"[WARNING] Skipping item in pack (image={item.get('image', 'N/A')}): {e}")
            if len(results) == 0:
                # Fallback: try a random different index
                return self.__getitem__(random.randint(0, len(self.data) - 1))
            return results
        else:
            # data[idx] is a single item
            item = self.data[idx]
            try:
                return self._process_single_item(item)
            except Exception as e:
                print(f"[WARNING] Skipping item (image={item.get('image', 'N/A')}): {e}")
                return self.__getitem__(random.randint(0, len(self.data) - 1))

    def _process_single_item(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process a single data item."""
        # Resolve image source: LMDB (multi or single) or filesystem.
        source = item.get("source")
        txn = (
            self._get_lmdb_txn(source)
            if (self.image_lmdb_root is not None or self.image_lmdb_path is not None)
            else None
        )

        if txn is not None:
            image_path = item.get("image", source or "<lmdb>")  # used for messages content; just informational
            image_id = item["image_id"]
            image_key = f"{image_id:08d}".encode()
            image_bytes = txn.get(image_key)
            if image_bytes is None:
                raise KeyError(f"image_id {image_id} not found in LMDB (source={source})")
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            image_path = item["image"]
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.image_folder, image_path)
            # Load image
            image = Image.open(image_path).convert("RGB")

        # Construct messages in the expected format
        messages = [
            {"role": "system", "content": item.get("system", "")},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": item["question"]},
                ],
            },
            {"role": "assistant", "content": item["answer"]},
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image,
            padding=False,
            return_tensors="pt",
        )

        # Prepare labels (same as input_ids for causal LM)
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Mask image tokens (ignore loss for image tokens)
        # image_token_id = 120120  # self.processor.image_token_id
        # image_token_mask = (input_ids == image_token_id)
        # labels[inputs["image_mask"][0]] = -100

        # Mask the prompt part (only compute loss on assistant's response)
        # Find the assistant token position
        assistant_text = self.processor.apply_chat_template(
            messages[:2],  # Only system and user
            tokenize=False,
            add_generation_prompt=True,
        )
        assistant_inputs = self.processor(
            text=[assistant_text],
            images=image,
            padding=False,
            return_tensors="pt",
        )
        prompt_length = assistant_inputs["input_ids"].shape[1]
        labels[:prompt_length] = -100  # Ignore loss for prompt tokens

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs.get("pixel_values", None),
            "image_grid_thw": inputs.get("image_grid_thw", None),
            "position_ids": inputs["position_ids"],
            "labels": labels,
        }


class PackedVLDataCollator:
    """
    Data collator that packs multiple samples into a single sequence.
    This is more efficient than padding, especially when samples have varying lengths.

    Key features:
    - Concatenates multiple samples into one sequence
    - Creates block-diagonal attention mask to isolate different samples
    - Generates correct position_ids for each sample (restarting from 0)
    - Handles image tokens correctly
    - Supports MTP (Multi-Token Prediction) extra data packing
    """

    def __init__(self, processor: HunYuanVLProcessor, packed_max_length: int = 2048):
        self.processor = processor
        self.max_length = packed_max_length
        # Robust pad_token_id resolution:
        # - If a multimodal Processor (e.g. HunYuanVLProcessor) was passed,
        #   read processor.tokenizer.pad_token_id.
        # - If a bare Tokenizer was passed by mistake, read its own pad_token_id.
        if hasattr(processor, "tokenizer"):
            self.pad_token_id = processor.tokenizer.pad_token_id
        elif hasattr(processor, "pad_token_id"):
            self.pad_token_id = processor.pad_token_id
        else:
            raise AttributeError(
                f"Cannot resolve pad_token_id from {type(processor).__name__}; "
                "expected a Processor with .tokenizer or a Tokenizer instance."
            )

    def _create_packed_attention_mask(self, sample_lengths: list[int], device: torch.device) -> torch.Tensor:
        """
        Create block-diagonal attention mask for packed sequences.
        Each sample can only attend to tokens within the same sample.

        Args:
            sample_lengths: List of lengths for each sample in the packed sequence
            device: Device to create the mask on

        Returns:
            Attention mask of shape [total_length, total_length]
        """
        total_length = sum(sample_lengths)
        # Create mask with 1s (allowed) and 0s (blocked)
        mask = torch.zeros((total_length, total_length), dtype=torch.bool, device=device)

        start_idx = 0
        for length in sample_lengths:
            end_idx = start_idx + length
            # Allow attention within this sample
            mask[start_idx:end_idx, start_idx:end_idx] = True
            start_idx = end_idx

        return mask

    def __call__(self, features: list[Any]) -> dict[str, torch.Tensor]:
        """
        Pack multiple samples into a single sequence.

        Args:
            features: Can be either:
                - List[Dict]: Normal unpacked data, each dict is a single sample
                - List[List[Dict]]: Packed data, each inner list is a pre-packed batch
        """
        # Handle packed data format: if features[0] is a list, it's already packed
        if isinstance(features[0], list):
            # Already packed: features = [[item1, item2, ...]]
            # We expect batch_size=1 for packed data, so features[0] is the packed batch
            features = features[0]

        # Now features is a list of dicts, process normally
        # Separate different types of inputs
        all_input_ids = [f["input_ids"] for f in features]
        all_labels = [f["labels"] for f in features]
        all_position_ids = [f["position_ids"] for f in features]

        # Pack sequences: original tokens first, MTP tokens appended at the end
        packed_input_ids = []  # Original sequence tokens
        packed_labels = []  # Original sequence labels
        packed_position_ids = []  # Original sequence position_ids
        sample_lengths = []

        current_length = 0
        packed_sample_indices = []  # Track which samples were actually packed
        for idx, (input_ids, labels, position_ids) in enumerate(zip(all_input_ids, all_labels, all_position_ids)):
            seq_len = len(input_ids)

            # Check if adding this sample would exceed max_length
            if current_length + seq_len > self.max_length:
                # If we haven't added any samples yet, truncate this one
                if current_length == 0:
                    packed_input_ids.append(input_ids[: self.max_length])
                    packed_labels.append(labels[: self.max_length])
                    packed_position_ids.append(position_ids[:, :, : self.max_length])
                    sample_lengths.append(self.max_length)
                    packed_sample_indices.append(idx)
                    current_length = self.max_length
                # Otherwise, stop packing
                break

            packed_input_ids.append(input_ids)
            packed_labels.append(labels)
            packed_position_ids.append(position_ids)
            sample_lengths.append(seq_len)
            packed_sample_indices.append(idx)
            current_length += seq_len

        # Concatenate original sequences
        packed_input_ids = torch.cat(packed_input_ids, dim=0)
        packed_labels = torch.cat(packed_labels, dim=0)
        packed_position_ids = torch.cat(packed_position_ids, dim=2)

        # Only collect pixel_values and image_grid_thw from samples that were actually packed
        # This prevents mismatch between image tokens in input_ids and image features
        all_pixel_values = [
            features[i]["pixel_values"] for i in packed_sample_indices if features[i]["pixel_values"] is not None
        ]
        all_image_grid_thw = [
            features[i]["image_grid_thw"] for i in packed_sample_indices if features[i]["image_grid_thw"] is not None
        ]

        # 构建attention_mask = cumsum_seq_lens, 使其能够用于flash attention
        # cu_seqlens only covers original sample lengths (MTP is handled separately)
        cumsum_seq_lens = torch.cumsum(torch.tensor([0] + sample_lengths), dim=0, dtype=torch.int32)
        attention_mask = cumsum_seq_lens

        # Prepare batch (add batch dimension)
        batch = {
            "input_ids": packed_input_ids.unsqueeze(0),  # [1, total_length]
            "attention_mask": attention_mask,  # cumsum_seq_lens for flash attention (original only)
            "position_ids": packed_position_ids,  # [1, 4, total_length]
            "labels": packed_labels.unsqueeze(0),  # [1, total_length]
        }

        # Handle image inputs
        if all_pixel_values:
            batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)

        if all_image_grid_thw:
            batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        # print('input_ids', batch['input_ids'].shape, 'image_mask', batch['image_mask'].shape)
        return batch


class VLDataCollator:
    """Custom data collator for vision-language data."""

    def __init__(self, processor: HunYuanVLProcessor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Separate different types of inputs
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        pixel_values = [f["pixel_values"] for f in features if f["pixel_values"] is not None]
        image_grid_thw = [f["image_grid_thw"] for f in features if f["image_grid_thw"] is not None]
        [f["position_ids"] for f in features]

        # Pad sequences
        max_len = min(max(len(ids) for ids in input_ids), self.max_length)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, label in zip(input_ids, attention_mask, labels):
            padding_length = max_len - len(ids)
            if padding_length > 0:
                padded_input_ids.append(
                    torch.cat([ids, torch.full((padding_length,), self.processor.tokenizer.pad_token_id)])
                )
                padded_attention_mask.append(torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)]))
                padded_labels.append(torch.cat([label, torch.full((padding_length,), -100)]))
            else:
                padded_input_ids.append(ids[:max_len])
                padded_attention_mask.append(mask[:max_len])
                padded_labels.append(label[:max_len])

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }

        if pixel_values:
            batch["pixel_values"] = torch.stack(pixel_values)
        if image_grid_thw:
            batch["image_grid_thw"] = torch.stack(image_grid_thw)

        return batch


if __name__ == "__main__":
    pass
