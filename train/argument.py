from dataclasses import dataclass, field

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default="./HunyuanOCR")
    siglip_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to SigLIP2 model directory (e.g. siglip2-base-patch16-naflex or siglip2-so400m-patch16-512). "
            "Used only when --from_scratch is True."
        },
    )
    tune_mm_llm: bool = field(default=True)
    tune_mm_mlp: bool = field(default=True)
    tune_mm_vision: bool = field(default=True)


@dataclass
class DataArguments:
    train_data_path: str = field(default="")
    eval_data_path: str | None = field(default=None, metadata={"help": "Path to the evaluation data (JSON format)"})
    image_folder: str = field(default="./data/images", metadata={"help": "Folder containing images"})
    image_lmdb_path: str = field(default=None)
    packed_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length"})
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    max_pixels: int = field(default=2048 * 2048)
    min_pixels: int = field(default=512 * 512)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str | None = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mm_projector_lr: float | None = None
    vision_tower_lr: float | None = None
    lr_scheduler_kwargs: dict | None = None
    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
    use_deepspeed: bool = field(default=True)
    from_scratch: bool = field(default=False)


@dataclass
class DraftArguments:
    """Extra arguments specific to MYDraft training."""

    num_draft_layers: int = field(
        default=5, metadata={"help": "Number of last LLM layers to copy as draft layers (K)."}
    )
    num_mask_tokens: int = field(
        default=16, metadata={"help": "Number of learnable mask tokens (N), i.e., max tokens predicted per step."}
    )
    only_draft: bool = field(default=False)
    use_kv_cache: bool = field(default=False)
    use_dflash_ori: bool = field(default=False)
    load_draft_path: str | None = field(default=None)
    use_distill: bool = field(default=False)
    loop_num: int = field(default=1)
    sample_block_num: int = field(default=16)
