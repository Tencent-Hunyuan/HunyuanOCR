"""
train_draft_from_dflash.py — Continue-training entry point for the DFlash
draft model, starting from an EXISTING DFlash checkpoint directory.

Differences from train_draft.py:
  * Adds a new CLI argument ``--dflash_init_dir`` (string, required) pointing
    to a directory containing ``config.json`` + ``model.safetensors`` of a
    previously-trained DFlash draft model.
  * Uses MYDraftFromDFlash (from dflash_draft_resume) instead of MYDraft so
    weights are loaded from that directory rather than copied from the last
    K layers of the (frozen) target model.
  * Everything else (target model loading, tokenizer, dataset, trainer, save
    logic) is identical to train_draft.py.

The original train_draft.py is left untouched.

Usage::

    torchrun ... train/train_draft_from_dflash.py \
        --model_name_or_path ./HunyuanOCR \
        --dflash_init_dir   /path/to/existing_dflash_ckpt_dir \
        --train_data_path   ./data/parsing_packed_20480.jsonl \
        ...
"""

import logging
import os
import pathlib
import sys

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import nn

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    HunYuanVLForConditionalGeneration,
    Trainer,
)

from train.argument import (
    DataArguments,
    DraftArguments,
    ModelArguments,
    TrainingArguments,
)
from train.data_processor import PackedVLDataCollator, VLDataset
from train.trainer import replace_hunyuanocr_attention_class

transformers.logging.set_verbosity_info()

local_rank = None
logging.basicConfig(level=logging.INFO, force=True)


# ---------------------------------------------------------------------------
# Extra CLI argument for the dflash checkpoint directory
# ---------------------------------------------------------------------------
@dataclass
class DFlashInitArguments:
    """Path to an existing DFlash checkpoint directory to initialize from.

    The directory must contain at least:
      - config.json          (matching DFlashDraftModel structure)
      - model.safetensors    (or pytorch_model.bin)
    """
    dflash_init_dir: str | None = field(
        default=None,
        metadata={
            "help": "Directory containing a previously-trained DFlash draft "
                    "checkpoint. If omitted, draft is randomly initialized "
                    "(NOT recommended). Conflicts with the default behavior "
                    "of copying weights from target's last K layers."
        },
    )


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ---------------------------------------------------------------------------
# Custom Trainer — identical to DraftOnlyTrainer in train_draft.py
# ---------------------------------------------------------------------------
class DraftOnlyTrainer(Trainer):
    """Surfaces ``ce_loss`` / ``distill_loss`` / ``accuracy`` into the standard
    HF logging pipeline."""

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        outputs = model(**inputs)
        loss_dict = outputs["loss"]
        loss = loss_dict["loss"]

        loss_dict_to_log = {
            k: (v.detach().float().mean().item() if isinstance(v, torch.Tensor) else float(v))
            for k, v in loss_dict.items()
        }
        self.log(loss_dict_to_log)

        if return_outputs:
            return loss, outputs
        return loss


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save only the draft parameters (matches train_draft.py)."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if not k.startswith("target_model.")
        }
        cpu_state_dict = {k: v.cpu() for k, v in draft_state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train(attn_implementation: str = "flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments,
         DraftArguments, DFlashInitArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        draft_args,
        dflash_init_args,
    ) = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # min_lr for cosine_with_min_lr scheduler. For finetune-from-v1 runs we
    # use a smaller min_lr (5e-7) so the late-stage LR decays more completely,
    # leaving the v1 weights largely intact in the final epochs.
    training_args.lr_scheduler_kwargs = {"min_lr": 5e-7}

    # ── Load processor / tokenizer ────────────────────────────────────────
    rank0_print("Loading processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    # Sanity check: in transformers 4.57, AutoProcessor may silently fall back
    # to a bare tokenizer if preprocessor_config.json is missing. We need the
    # full multimodal processor for image handling.
    if not hasattr(processor, "tokenizer"):
        raise RuntimeError(
            f"AutoProcessor.from_pretrained({model_args.model_name_or_path}) "
            f"returned {type(processor).__name__} which is not a multimodal "
            "Processor. Check that preprocessor_config.json exists in the "
            "model directory."
        )

    # ── Load target model (frozen base) ───────────────────────────────────
    rank0_print(f"Loading target model from {model_args.model_name_or_path} ...")
    target_model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=attn_implementation,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    )

    # Replace attention class for packed / flatten training
    replace_hunyuanocr_attention_class()
    target_model.config.use_cache = False
    target_model.config.block_size = draft_args.num_mask_tokens
    target_model.config.num_draft_layers = draft_args.num_draft_layers

    # ── Build MYDraft, initialized from external dflash checkpoint ────────
    from train.dflash_draft_resume import MYDraftFromDFlash

    if dflash_init_args.dflash_init_dir is None:
        rank0_print(
            "[WARNING] --dflash_init_dir not specified. Draft will be randomly "
            "initialized. If you intended to *fine-tune from a base model* "
            "(copy weights from target's last K layers), use train_draft.py "
            "instead of train_draft_from_dflash.py."
        )

    rank0_print(
        f"Initializing draft model from dflash directory: "
        f"{dflash_init_args.dflash_init_dir}"
    )
    model = MYDraftFromDFlash(
        config=target_model.config,
        target_model=target_model,
        dflash_init_dir=dflash_init_args.dflash_init_dir,
        use_distill=draft_args.use_distill,
        loop_num=draft_args.loop_num,
        sample_block_num=draft_args.sample_block_num,
    )
    model.config.use_cache = False

    # Optional: legacy --load_draft_path is still respected (overrides
    # dflash_init_dir's weights). Useful for resume-from-best-ckpt scenarios.
    if draft_args.load_draft_path is not None:
        from safetensors.torch import load_file as safetensors_load_file
        draft_state_dict = safetensors_load_file(
            draft_args.load_draft_path, device="cpu"
        )
        draft_state_dict_new = {}
        for k, v in draft_state_dict.items():
            if k.startswith("draft_model."):
                draft_state_dict_new[k[len("draft_model."):]] = v
            else:
                draft_state_dict_new[k] = v
        model.load_state_dict(draft_state_dict_new, strict=False)
        rank0_print(
            f"Additional --load_draft_path applied: "
            f"{draft_args.load_draft_path}"
        )

    if local_rank == 0:
        model.print_parameter_info()

    # ── Datasets ──────────────────────────────────────────────────────────
    rank0_print("Loading datasets...")
    train_dataset = VLDataset(
        data_path=data_args.train_data_path,
        image_folder=data_args.image_folder,
        image_lmdb_path=data_args.image_lmdb_path,
        processor=processor,
        max_length=data_args.packed_max_length,
        is_packed=data_args.data_flatten or data_args.data_packing,
    )

    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = VLDataset(
            data_path=data_args.eval_data_path,
            image_folder=data_args.image_folder,
            processor=processor,
            max_length=data_args.packed_max_length,
        )

    data_collator = PackedVLDataCollator(
        processor=processor,
        packed_max_length=data_args.packed_max_length,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = DraftOnlyTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logger.info("Checkpoint found, resuming training...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
