"""
dflash_draft_resume.py — Variant of dflash_draft.MYDraft that supports
**initializing the draft model from an existing DFlash checkpoint directory**
(e.g. a previously released DFlash checkpoint).

Differences from the original ``MYDraft``:
  1. ``__init__`` no longer copies weights from the last K target layers of the
     base model. Instead, it loads draft weights from
     ``dflash_init_dir/model.safetensors`` (if provided).
  2. The draft config (``config_dflash``) is loaded from ``dflash_init_dir``
     when provided, otherwise it falls back to the bundled draft config
     template at ``train/configs/`` (overridable via the env var
     ``HYOCR_DFLASH_CONFIG_DIR``).
  3. Strict mismatch handling: prints any missing / unexpected keys so you can
     see whether the load was clean.

Everything else (forward pass, loss, etc.) is reused unchanged via subclass.

Usage (in train_draft_from_dflash.py)::

    from train.dflash_draft_resume import MYDraftFromDFlash
    model = MYDraftFromDFlash(
        config=target_model.config,
        target_model=target_model,
        dflash_init_dir="/path/to/dflash_checkpoint_dir",
        loop_num=draft_args.loop_num,
        sample_block_num=draft_args.sample_block_num,
        use_distill=draft_args.use_distill,
    )
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig

# Reuse the original implementation
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train.dflash_draft import DFlashDraftModel, MYDraft


class MYDraftFromDFlash(MYDraft):
    """MYDraft variant that initializes draft weights from an existing
    DFlash checkpoint directory instead of copying from target's last layers.
    """

    def __init__(
        self,
        config,
        target_model: nn.Module,
        dflash_init_dir: str | None = None,
        num_draft_layers: int = 5,
        only_draft: bool = False,
        use_distill: bool = False,
        loop_num: int = 1,
        sample_block_num: int = 16,
    ):
        # ── 1. Manually replicate the parent's __init__ but with two changes:
        #       (a) Load draft config from dflash_init_dir if provided.
        #       (b) Skip _initialize_draft_from_target (we'll load weights
        #           from a checkpoint instead).
        # We can't simply call super().__init__() because it copies from
        # target layers and we want to avoid that work entirely.

        # Go up two levels in the MRO: skip MYDraft.__init__, call its parent
        # (Qwen3PreTrainedModel.__init__) directly.
        # Equivalently: invoke nn.Module's grandparent init via PreTrainedModel.
        from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel

        Qwen3PreTrainedModel.__init__(self, config)

        self.config = config
        self.loss_decay_gamma = 7.0
        self.use_distill = use_distill
        self.loop_num = loop_num
        self.sample_block_num = sample_block_num
        num_draft_layers = self.config.num_draft_layers

        # Target model (frozen) — same as original
        self.target_model = target_model
        for p in self.target_model.parameters():
            p.requires_grad = False
        self.target_model.eval()

        hidden_size = config.hidden_size
        self.hidden_size = hidden_size

        # ── 2. Load DFlash config from dflash_init_dir (if given), or fallback
        #       to the bundled draft config template at train/configs/.
        #       The env var HYOCR_DFLASH_CONFIG_DIR overrides the fallback.
        if dflash_init_dir is not None and os.path.isdir(dflash_init_dir):
            cfg_dir = dflash_init_dir
            print(f"[MYDraftFromDFlash] Loading dflash config from: {cfg_dir}")
        else:
            cfg_dir = os.environ.get(
                "HYOCR_DFLASH_CONFIG_DIR",
                str(Path(__file__).parent / "configs"),
            )
            print(f"[MYDraftFromDFlash] dflash_init_dir not given, falling back to template config dir: {cfg_dir}")

        config_dflash = AutoConfig.from_pretrained(cfg_dir, trust_remote_code=True)
        config_dflash._attn_implementation = "flex_attention"
        config_dflash.num_hidden_layers = num_draft_layers
        config_dflash.block_size = config.block_size
        self.draft_model = DFlashDraftModel(config_dflash)

        # MTP configuration
        self.block_size = config.block_size
        self.mask_token_id = getattr(config, "mask_token_id", 120817)
        self.only_draft = only_draft

        # ── 3. Load draft weights from the dflash checkpoint (if provided)
        if dflash_init_dir is not None:
            self._load_draft_weights_from_dir(dflash_init_dir)
        else:
            print(
                "[MYDraftFromDFlash] No dflash_init_dir, draft weights "
                "are randomly initialized (you probably don't want this)."
            )

    # ------------------------------------------------------------------
    # Draft weight loader
    # ------------------------------------------------------------------
    def _load_draft_weights_from_dir(self, dflash_dir: str):
        """Load draft_model parameters from a DFlash checkpoint directory.

        The directory should contain ``model.safetensors`` (preferred) or
        ``pytorch_model.bin``. State-dict keys may or may not have the
        ``draft_model.`` prefix; both are handled.
        """
        # Locate the weight file
        candidates = [
            os.path.join(dflash_dir, "model.safetensors"),
            os.path.join(dflash_dir, "pytorch_model.bin"),
        ]
        weight_path = next((p for p in candidates if os.path.isfile(p)), None)
        if weight_path is None:
            raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {dflash_dir}")

        print(f"[MYDraftFromDFlash] Loading draft weights from: {weight_path}")

        # Load state dict
        if weight_path.endswith(".safetensors"):
            from safetensors.torch import load_file as safetensors_load_file

            raw_state_dict = safetensors_load_file(weight_path, device="cpu")
        else:
            raw_state_dict = torch.load(weight_path, map_location="cpu")

        # Normalize keys: strip optional "draft_model." prefix so they match
        # self.draft_model.<param_name>.
        normalized = {}
        for k, v in raw_state_dict.items():
            if k.startswith("draft_model."):
                normalized[k[len("draft_model.") :]] = v
            elif k.startswith("target_model."):
                # Should not appear in a draft-only checkpoint, but skip if so.
                continue
            else:
                normalized[k] = v

        missing, unexpected = self.draft_model.load_state_dict(normalized, strict=False)

        # Report load result on rank 0 only (best-effort).
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            rank = 0

        if rank == 0:
            print("[MYDraftFromDFlash] State-dict load complete:")
            print(f"  - Source keys                : {len(raw_state_dict)}")
            print(f"  - After prefix normalization : {len(normalized)}")
            print(f"  - Missing in draft_model     : {len(missing)}")
            print(f"  - Unexpected from checkpoint : {len(unexpected)}")
            if missing:
                preview = missing[:10]
                print(f"    missing (first 10): {preview}")
                if len(missing) > 10:
                    print(f"    ... and {len(missing) - 10} more")
            if unexpected:
                preview = unexpected[:10]
                print(f"    unexpected (first 10): {preview}")
                if len(unexpected) > 10:
                    print(f"    ... and {len(unexpected) - 10} more")

            if not missing and not unexpected:
                print("  [OK] Clean load.")
            else:
                print("  [WARN] State-dict mismatch detected — verify the checkpoint matches the current architecture.")
