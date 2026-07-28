"""
train_draft.py — Training script for MYDraft speculative decoding model.

Usage:
    torchrun --nproc_per_node=8 train/train_draft.py \
        --model_name_or_path ./HunyuanOCR \
        --train_data_path ./data/train.json \
        --output_dir ./output/draft \
        --data_packing True \
        --mtp_group 8 \
        --mtp_only_labels True \
        --num_draft_layers 5 \
        --num_mask_tokens 64 \
        --bf16 True \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --save_steps 500 \
        --logging_steps 10
"""

import logging
import os
import pathlib
import sys

logger = logging.getLogger(__name__)
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

# from hunyuan_vl import HunYuanVLForConditionalGeneration
# from hunyuan_vl.processing_hunyuan_vl import HunYuanVLProcessor
# from hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor
# from hunyuan_vl.configuration_hunyuan_vl import HunYuanVLConfig



local_rank = None

logging.basicConfig(level=logging.INFO, force=True)





def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ============================================================================
# Custom Trainer: capture per-component scalars from MYDraft.forward and feed
# them through Trainer.log() so they get routed to ALL configured reporters
# (swanlab, tensorboard, wandb, ...) automatically via TrainerCallback.
# ============================================================================
class DraftOnlyTrainer(Trainer):
    """Trainer that surfaces ``ce_loss`` / ``distill_loss`` / ``accuracy``
    (returned by ``MYDraft.forward`` as a dict in ``outputs["loss"]``) into
    the standard HF logging pipeline (swanlab / tensorboard / wandb / ...).

    ``MYDraft.forward`` now returns a ``CausalLMOutputWithPast`` whose
    ``loss`` field is itself a ``dict``::

        {
            "loss":         <main scalar used for backprop>,
            "ce_loss":      <cross-entropy scalar>,
            "distill_loss": <distillation scalar>,
            "accuracy":     <token-level accuracy>,
        }

    We unpack that dict here, push every entry through ``self.log(...)``
    so that all configured reporters (``--report_to swanlab tensorboard``)
    see them, and return the main scalar to Trainer for backprop.
    """

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        # Forward pass — outputs["loss"] is a dict of scalars.
        outputs = model(**inputs)
        loss_dict = outputs["loss"]

        # Main loss used for backprop.
        loss = loss_dict["loss"]

        # Build a logging-only dict (detached, on CPU as python floats).
        loss_dict_to_log = {
            k: (v.detach().float().mean().item() if isinstance(v, torch.Tensor) else float(v))
            for k, v in loss_dict.items()
        }
        # Push every sub-loss / accuracy into the standard HF logging
        # pipeline so swanlab / tensorboard / wandb all pick them up.
        self.log(loss_dict_to_log)

        if return_outputs:
            return loss, outputs
        return loss


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save only the trainable draft parameters."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        # Only save draft-related parameters (exclude frozen target_model)
        draft_state_dict = {
            k.replace("draft_model.", ""): v for k, v in state_dict.items()
            if not k.startswith("target_model.")
        }
        cpu_state_dict = {k: v.cpu() for k, v in draft_state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, DraftArguments)
    )
    model_args, data_args, training_args, draft_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    training_args.lr_scheduler_kwargs = {'min_lr': 2e-6}

    # ── Load processor ────────────────────────────────────────────────────
    rank0_print("Loading processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
    )

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )
    

    # ── Load target model ─────────────────────────────────────────────────
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

    from train.dflash_draft import MYDraft

    model = MYDraft(
        config=target_model.config,
        target_model=target_model,
        use_distill=draft_args.use_distill,
        loop_num=draft_args.loop_num,
        sample_block_num=draft_args.sample_block_num,
    )
    model.config.use_cache = False

    if draft_args.load_draft_path is not None:
        # load safe checkpoint
        from safetensors.torch import load_file as safetensors_load_file
        draft_state_dict = safetensors_load_file(draft_args.load_draft_path, device="cpu")
        # draft_state_dict = torch.load(model_args.load_draft_path, map_location="cpu")
        draft_state_dict_new = {}
        for k, v in draft_state_dict.items():
            if k.startswith("draft_model."):
                draft_state_dict_new[k[len("draft_model."):]] = v
            else:
                draft_state_dict_new[k] = v
                # del draft_state_dict[k]
        model.load_state_dict(draft_state_dict_new, strict=False)
        rank0_print(f"Loading draft model from {draft_args.load_draft_path} successfully")

    # model.load_state_dict()

    # gradient_checkpointing is handled inside MYDraft.gradient_checkpointing_enable()
    # which disables checkpointing on draft_layers (monkey-patched, incompatible).
    # No extra hook needed here.

    if local_rank == 0:
        model.print_parameter_info()

    # ── Load datasets ─────────────────────────────────────────────────────
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

    # ── Data collator ─────────────────────────────────────────────────────

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
