# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import os
import pathlib
import sys

logger = logging.getLogger(__name__)
from pathlib import Path

import torch
import transformers

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent  # 指向 hunyuan_vl_finetune 目录
sys.path.insert(0, str(project_root))

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    HunYuanVLForConditionalGeneration,
    Trainer,
)

from train.argument import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from train.data_processor import PackedVLDataCollator, VLDataCollator, VLDataset
from train.trainer import replace_hunyuanocr_attention_class

transformers.logging.set_verbosity_info()


local_rank = None

logging.basicConfig(level=logging.INFO, force=True)

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.vit.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.vit.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.vit.perceive.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.vit.perceive.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank



    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    

    local_rank = training_args.local_rank
    # if not training_args.use_deepspeed:
    #     torch.distributed.init_process_group(backend='nccl')
    #     torch.cuda.set_device(torch.device(f'cuda:{local_rank}'))
    os.makedirs(training_args.output_dir, exist_ok=True)

    training_args.lr_scheduler_kwargs = {'min_lr': 2e-6}
    # Load processor
    rank0_print("Loading processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )
    
    # Load model
    rank0_print(f"Loading model... with bf16 type: {training_args.bf16}")
    if training_args.from_scratch:
        config_path = model_args.model_name_or_path
        from transformers.models.hunyuan_vl.configuration_hunyuan_vl import (
            HunYuanVLConfig,
        )
        config = HunYuanVLConfig.from_pretrained(config_path)
        config._attn_implementation = attn_implementation
        model = HunYuanVLForConditionalGeneration(config)
        model = model.to(torch.bfloat16 if training_args.bf16 else torch.float32)
    else:
        model = HunYuanVLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=attn_implementation,
            # attn_implementation="eager",
            dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
            trust_remote_code=True
        )

    if data_args.data_flatten or data_args.data_packing:
        replace_hunyuanocr_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, TaskType, get_peft_model
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.vit.print_trainable_parameters()
            model.model.print_trainable_parameters()

    # data_module = make_supervised_data_module(processor, data_args=data_args)
    # Load datasets
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
            max_length=data_args.packed_max_length
        )
    
    # Data collator - choose based on whether packing is enabled
    if data_args.data_flatten or data_args.data_packing:
        rank0_print("Using packed data collator for efficient training...")
        data_collator = PackedVLDataCollator(processor=processor, packed_max_length=data_args.packed_max_length)
    else:
        rank0_print("Using standard data collator with padding...")
        data_collator = VLDataCollator(processor=processor, max_length=data_args.packed_max_length)

    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logger.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
