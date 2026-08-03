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
from transformers.models.hunyuan_vl.modeling_hunyuan_vl import HunYuanVLModel


class _RopeIndexShim:
    """Lightweight carrier that reuses the model's exact multimodal position-id logic.

    ``HunYuanVLModel.get_rope_index`` / ``get_vision_position_ids`` depend only on
    ``self.config`` (no weights, no submodules), so we bind them to a shim holding just
    the config. This guarantees the packed collator's per-sample position_ids match the
    model's own convention without instantiating the full (multi-billion parameter) model.
    """

    get_rope_index = HunYuanVLModel.get_rope_index
    get_vision_position_ids = HunYuanVLModel.get_vision_position_ids

    def __init__(self, config):
        self.config = config


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
        model_config: Any = None,
    ):
        super().__init__()
        self.processor = processor
        self.max_length = max_length
        self.image_folder = image_folder
        self.is_packed = is_packed
        # For packed training we must emit per-sample multimodal position_ids that restart
        # at each sample boundary. We reuse the model's own get_rope_index via a config-only
        # shim so the layout matches the model exactly.
        self._rope_shim = _RopeIndexShim(model_config) if (is_packed and model_config is not None) else None

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
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )

        # Prepare labels (same as input_ids for causal LM)
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Image placeholders and the prompt are context, not prediction targets.
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is not None:
            labels[input_ids == image_token_id] = -100

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
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )
        prompt_length = assistant_inputs["input_ids"].shape[1]
        labels[:prompt_length] = -100  # Ignore loss for prompt tokens

        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None and mm_token_type_ids.shape[0] == 1:
            mm_token_type_ids = mm_token_type_ids[0]

        # For packed training, precompute this sample's multimodal 3D position_ids using the
        # model's own get_rope_index (via the config-only shim). These positions start at 0 for
        # every sample; concatenating them in the packed collator — together with a block-diagonal
        # causal mask — reproduces the varlen/packed attention semantics natively.
        position_ids = None
        if self._rope_shim is not None:
            rope_positions, _ = self._rope_shim.get_rope_index(
                input_ids=inputs["input_ids"],
                mm_token_type_ids=inputs["mm_token_type_ids"],
                image_grid_thw=inputs.get("image_grid_thw"),
                attention_mask=inputs["attention_mask"],
            )
            # rope_positions: [num_mrope_axes, batch=1, seq_len] -> drop batch dim
            position_ids = rope_positions[:, 0, :]

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "position_ids": position_ids,
            "mm_token_type_ids": mm_token_type_ids,
            "labels": labels,
        }


class PackedVLDataCollator:
    """
    Data collator that packs multiple samples into a single sequence.
    This is more efficient than padding, especially when samples have varying lengths.

    Packing is expressed entirely through inputs the native HunYuanVL forward already
    understands — no attention monkey-patching:
    - Concatenates the packed samples into one [1, total_len] sequence.
    - Emits a block-diagonal additive causal mask [1, 1, L, L] so each sample only attends
      within itself (varlen semantics).
    - Concatenates each sample's multimodal 3D position_ids (each restarting from 0), which
      the dataset precomputes via the model's own get_rope_index.
    - Merges visual inputs as flat pixel_values / [num_images, 3] image_grid_thw.
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

    def _create_packed_causal_mask(self, sample_lengths: list[int], dtype: torch.dtype) -> torch.Tensor:
        """Block-diagonal causal mask for a packed sequence.

        Returns an additive mask of shape ``[1, 1, L, L]`` where, within each sample block,
        a token may attend to itself and earlier tokens of the same sample, and everything
        outside its own block is masked with ``-inf``. This is exactly the varlen/packed
        attention pattern, expressed in the form the native HunYuanVL forward consumes.
        """
        total_length = sum(sample_lengths)
        min_val = torch.finfo(dtype).min
        mask = torch.full((total_length, total_length), min_val, dtype=dtype)
        start_idx = 0
        for length in sample_lengths:
            end_idx = start_idx + length
            block = torch.triu(
                torch.full((length, length), min_val, dtype=dtype), diagonal=1
            )
            mask[start_idx:end_idx, start_idx:end_idx] = block
            start_idx = end_idx
        return mask.unsqueeze(0).unsqueeze(0)

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

        all_input_ids = [f["input_ids"] for f in features]
        all_labels = [f["labels"] for f in features]
        all_position_ids = [f["position_ids"] for f in features]

        packed_input_ids = []
        packed_labels = []
        packed_position_ids = []  # each entry is [num_mrope_axes, seq_len]
        sample_lengths = []

        current_length = 0
        packed_sample_indices = []
        for idx, (input_ids, labels, position_ids) in enumerate(zip(all_input_ids, all_labels, all_position_ids)):
            if position_ids is None:
                raise ValueError(
                    "PackedVLDataCollator requires per-sample position_ids. The processor did not "
                    "return position_ids for a sample; ensure the HunYuanVL processor emits them."
                )
            seq_len = len(input_ids)

            if current_length + seq_len > self.max_length:
                if current_length == 0:
                    packed_input_ids.append(input_ids[: self.max_length])
                    packed_labels.append(labels[: self.max_length])
                    packed_position_ids.append(position_ids[:, : self.max_length])
                    sample_lengths.append(self.max_length)
                    packed_sample_indices.append(idx)
                    current_length = self.max_length
                break

            packed_input_ids.append(input_ids)
            packed_labels.append(labels)
            packed_position_ids.append(position_ids)
            sample_lengths.append(seq_len)
            packed_sample_indices.append(idx)
            current_length += seq_len

        packed_input_ids = torch.cat(packed_input_ids, dim=0)
        packed_labels = torch.cat(packed_labels, dim=0)
        # position_ids per sample are [num_mrope_axes, seq_len]; concat along the sequence axis.
        # Each sample already restarts positions from 0, which — combined with the block-diagonal
        # mask below — reproduces the varlen/packed attention semantics natively.
        packed_position_ids = torch.cat(packed_position_ids, dim=-1).unsqueeze(1)  # [axes, 1, total_len]

        all_pixel_values = [
            features[i]["pixel_values"] for i in packed_sample_indices if features[i]["pixel_values"] is not None
        ]
        all_image_grid_thw = [
            features[i]["image_grid_thw"] for i in packed_sample_indices if features[i]["image_grid_thw"] is not None
        ]

        attention_mask = self._create_packed_causal_mask(sample_lengths, dtype=torch.float32)

        batch = {
            "input_ids": packed_input_ids.unsqueeze(0),  # [1, total_length]
            "attention_mask": attention_mask,  # block-diagonal additive causal mask [1, 1, L, L]
            "position_ids": packed_position_ids,  # [num_mrope_axes, 1, total_length]
            "labels": packed_labels.unsqueeze(0),  # [1, total_length]
        }

        if all_pixel_values:
            batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grid_thw:
            batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        return batch


class VLDataCollator:
    """Custom data collator for vision-language data."""

    def __init__(self, processor: HunYuanVLProcessor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        max_len = min(max(ids.shape[-1] for ids in input_ids), self.max_length)
        pad_token_id = self.processor.tokenizer.pad_token_id

        def pad_sequence(tensor: torch.Tensor, value: int | bool) -> torch.Tensor:
            tensor = tensor[..., :max_len]
            padding_length = max_len - tensor.shape[-1]
            if padding_length == 0:
                return tensor
            padding_shape = (*tensor.shape[:-1], padding_length)
            padding = torch.full(padding_shape, value, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=-1)

        batch = {
            "input_ids": torch.stack([pad_sequence(ids, pad_token_id) for ids in input_ids]),
            "attention_mask": torch.stack([pad_sequence(mask, False) for mask in attention_mask]),
            "labels": torch.stack([pad_sequence(label, -100) for label in labels]),
        }

        optional_token_fields = ("position_ids", "mm_token_type_ids")
        for field in optional_token_fields:
            values = [f.get(field) for f in features]
            if all(value is not None for value in values):
                batch[field] = torch.stack([pad_sequence(value, 0) for value in values])

        pixel_values = [f["pixel_values"] for f in features if f.get("pixel_values") is not None]
        image_grid_thw = [f["image_grid_thw"] for f in features if f.get("image_grid_thw") is not None]
        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        if image_grid_thw:
            batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        return batch


if __name__ == "__main__":
    pass
