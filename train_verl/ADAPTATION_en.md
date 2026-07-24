# HunyuanOCR-1.5 · verl Adaptation Notes

> 🌏 **中文版**: [ADAPTATION.md](./ADAPTATION.md)

This document catalogs every change we made to [verl](https://github.com/volcengine/verl) while adapting the framework for GRPO training of HunyuanOCR-1.5, together with the hyperparameters that are frozen in `train_grpo.sh` and should not be moved carelessly.

Upstream baseline: verl official `main` branch, commit [`2b47a68`](https://github.com/volcengine/verl/commit/2b47a68b66d6fa21884990a5d4445ceba59c47e5) (2026-07-24 snapshot). All changes live in the personal fork: [pspdada/verl-HYOCR · branch `HYOCR`](https://github.com/pspdada/verl-HYOCR/tree/HYOCR).

The changes fall into three categories:

- [A. HunyuanOCR-specific RL adaptations](#a-hunyuanocr-specific-rl-adaptations): tightly coupled to the model architecture, processor, xdrope, and weight naming. Miss any one and HunyuanOCR simply cannot run.
- [B. General functional adaptations](#b-general-functional-adaptations): non-HunyuanOCR-specific enhancements; other models benefit too.
- [C. Mandatory hyperparameter choices](#c-mandatory-hyperparameter-choices): switches hardcoded in the training script that cannot be flipped casually today.

---

## A. HunyuanOCR-specific RL adaptations

### A1. Collapse expanded multimodal placeholders after rollout

**File**: `verl/experimental/agent_loop/agent_loop.py`

Added a helper `_collapse_expanded_mm_placeholders(text, processor)` and replaced the single `tokenizer.decode(..., skip_special_tokens=True)` call inside `AgentLoopWorker` with `skip_special_tokens=False` + collapse.

Why: by the end of rollout `input_ids` has already been expanded by the HF processor (HunYuanVL turns each `<img_start><ph><img_end>` into `<img_start><ph>*N<img_end>` with N equal to the number of visual patches). Feeding a naive `tokenizer.decode` output back into the processor would fail HF's `validate_inputs` check (which requires "placeholder token count == number of images") in two ways:

- `skip_special_tokens=True` → all placeholders stripped, count = 0;
- `skip_special_tokens=False` → count = N × num_patches, far exceeding num_images.

Collapsing restores each run of consecutive placeholders back to a single one, so the processor can re-expand once using `image_grid_thw`.

**Side effects**: none. The helper guards on three attributes via `getattr(processor, "image_start_token", None)` etc.; non-VLM or processors without these attributes are returned unchanged.

---

### A2. Do not bind `get_rope_index` to `HunYuanVLProcessor`

**File**: `verl/utils/tokenizer/tokenizer.py`

Added a dedicated branch in `hf_processor()`'s processor `case`:

```python
case "HunYuanVLProcessor":
    pass  # keep processor untouched; do not inject get_rope_index
```

Why: HunYuanVL uses xdrope (4-way RoPE with an `xdrope_section` split); the Qwen2VL 3D `(T, H, W)` implementation is incompatible with xdrope, and HF's `HunYuanVLModel` itself does not expose `get_rope_index`. The upper layer checks via `hasattr(processor, "get_rope_index")`; when absent, it falls back to a 1D `torch.arange` for position_ids, matching how HunyuanOCR is actually trained.

**Side effects**: none. New standalone case; other processors follow the original path unchanged.

---

### A3. Map HunyuanVL weight keys to vLLM naming

**File**: `verl/utils/model.py`

Two changes:

1. **Add a HunyuanVL weight-prefix rewrite inside `convert_weight_keys`**. Guard: `if any(k.startswith("model.language_model.") or k.startswith("model.vision_tower.") for k in state_dict)`. Once triggered, a two-stage regex rewrite runs:
   - **Stage 1 (top-level prefixes)**: `model.language_model.*` → `model.*`; `model.vision_tower.*` → `vit.*`, aligning to the layout that vLLM's `HunYuanVLForConditionalGeneration.hf_to_vllm_mapper` expects.
   - **Stage 2 (vision internals)**: reconcile naming differences between vLLM's `HunYuanVisionTransformer` and the HF side, including `patch_embed → embeddings`, `layer_norm1/2 → input/post_attention_layernorm`, `mlp.fc1/2 → dense_h_to_4h/dense_4h_to_h`, `patch_merger.proj_conv/proj_out → perceive.proj.0/2`, etc. (`q/k/v_proj → qkv` is handled by vLLM's own `stacked_params_mapping` and is not done here.)

2. **Add an `AutoModelForVision2Seq` fallback to `_architecture_to_auto_class` / `get_hf_auto_model_class`**. `HunYuanVLForConditionalGeneration` is only registered in `AutoModelForVision2Seq._model_mapping`; without this fallback verl cannot recognise or load it.

**Side effects**: none. The rewrite guard explicitly targets HunyuanVL-specific prefixes; state_dicts of other models don't carry these prefixes so the whole block is skipped. `AutoModelForVision2Seq` is only added as a fallback path and does not override existing priorities.

---

### A4. Runtime patches for HunyuanVL after FSDP wrapping

**File**: `verl/workers/engine/fsdp/transformer_impl.py`

After FSDP wraps the model, `_patch_vlm_get_image_features(module)` is invoked to do three things:

1. **Fix `StopIteration` in `get_image_features` / `get_video_features`**. HF's implementations probe vit dtype via `next(self.vit.parameters()).dtype`; after FSDP flattens parameters `self.vit.parameters()` becomes an empty iterator, so `next(...)` raises. The patch reads `engine_config.model_dtype` instead (defaults to bf16).

2. **Rewrite `HunYuanVLModel.forward` so 3D `position_ids` are compatible with `create_causal_mask`**. xdrope's position_ids have shape `(bs, 4, seq)`, while `transformers.masking_utils.create_causal_mask` only accepts 2D. The patch slices `position_ids[:, -1, :]` (xdrope's 4th plane = standard text position) before handing off to `create_causal_mask`; decoder layers still receive the full 3D tensor for `apply_rotary_pos_emb_xdrope`.

3. **Fix the 3D position_ids layout in `FSDPEngineWithLMHead.forward`**. The original implementation transposed `(bs, 4, seq)` into `(4, bs, seq)` via `.transpose(0, 1)`; xdrope's `apply_rotary_pos_emb_xdrope` expects `(bs, 4, seq)`, so the transpose is removed.

**Side effects**: none. The method-level patch is guarded by `hasattr(unwrapped, "vit")` and `hasattr(text_model, "layers")`; the 3D layout fix inside `forward` only fires under `if position_ids.dim() == 3`, and other models continue through the 2D branch unchanged.

---

### A5. Collapse already-expanded image / video placeholders at rollout time

**File**: `verl/workers/rollout/utils.py`

`qwen2_5_vl_dedup_image_tokens` widens its supported list from a single `Qwen2VLImageProcessor` to `("Qwen2VLImageProcessor", "HunYuanVLImageProcessor")`, and enables the collapse for any processor exposing a non-empty `image_token_id`.

Why: the `input_ids` returned by HF `processor.apply_chat_template(...)` already has each `<image>` expanded into N consecutive `image_token_id`s (N = number of visual patches). The vLLM engine expects "exactly one placeholder per image" and expands them itself once more. If both sides expand, every already-expanded token in vLLM gets a distinct M/XD-RoPE position id, visual features land at wrong rotary positions, and the model produces fluent output that is completely image-blind (the symptom observed on HunyuanOCR-1.5).

**Side effects**: Qwen2VL is unchanged (it's in the supported list). For any new VLM processor exposing `image_token_id`, the dedup branch is **newly taken**. This is semantically correct: if the VLM also pre-expands placeholders, dedup is required; if it does not, `is_value` is all False, `mask` keeps everything, and dedup is idempotent.

---

### A6. Force the `auto` weight-load path when vLLM loads a VLM

**File**: `verl/workers/rollout/vllm_rollout/vllm_async_server.py`

When `vLLMHttpServer` detects `hf_config` contains `vision_config` and `load_format == "dummy"`, it forces the load format back to `auto`.

Why: a VLM's ViT / M-RoPE / mm_receiver contain non-tensor state that is only initialised on the `auto` load path; taking the dummy route skips these initialisations, and rollout output degenerates into an image-blind repeating token stream.

**Side effects**: only triggers on models that expose `vision_config`; text-only models continue using the dummy path unchanged.

---

## B. General functional adaptations

### B1. Thread the `use_fast_processor` switch through the whole pipeline

**Involves**: `verl/experimental/fully_async_policy/fully_async_main.py`, `verl/experimental/one_step_off_policy/main_ppo.py`, `verl/trainer/main_ppo_v0.py`, `verl/trainer/ppo/v1/trainer_base.py`, `verl/workers/config/model.py`, `verl/trainer/config/model/hf_model.yaml`.

The hardcoded `hf_processor(name_or_path, ..., use_fast=True)` is switched to read `config.data.use_fast_processor` / `HFModelConfig.use_fast_processor` (default True), and the value is injected into `mm_processor_kwargs.use_fast` when vLLM starts up, so that the image preprocessing backend on the rollout side and the training side stays consistent.

### B2. RLLoggingBoard training-log dump

**Involves**: `verl/utils/logger/rl_logging_board_writer.py` (new), `verl/trainer/ppo/ray_trainer.py`, `verl/trainer/ppo/v1/trainer_base.py`.

Added `dump_batch_for_rl_logging_board(...)`: converts a single training step's `DataProto` batch into per-sample JSONL matching the RLLoggingBoard UI, with fields such as `prompt / response / reward / logprobs / ref_logprobs / values / advantages / kl / reward_tokens / img_path / data_source / task_type`. `_log_rollout_data` is split from a single sink into two independent channels ("built-in `{step}.jsonl`" and "RLLoggingBoard writer"). `_compute_reward_colocate` persists the reward manager's `reward_extra_keys` into `batch.extra_info` so downstream dumps can read them.

### B3. Disk cache for filter-processed datasets

**File**: `verl/utils/dataset/rl_dataset.py`

Added three config options `enable_dataset_cache` / `dataset_cache_dir` / `dataset_cache_force_rebuild`. The cache key is an md5 fingerprint of tokenizer name + data_files + filter arguments. On a hit, `datasets.load_from_disk` returns immediately; otherwise after `.filter(...)` completes, `save_to_disk` writes to `filtered_{hash}/` (first as `.tmp`, then renamed, so an interrupted run cannot leave a half-baked cache).

### B4. Multi-replica vLLM cold-start acceleration

**Involves**: `verl/workers/rollout/llm_server.py`, `verl/workers/rollout/vllm_rollout/vllm_async_server.py`, `verl/workers/rollout/vllm_rollout/utils.py`.

In a multi-replica deployment only replica 0 loads real weights from disk; the rest boot with `force_load_format="dummy"`, and weights are streamed to followers in 500 MB chunks via `/dev/shm` + the Ray object store. New async helpers were added to coordinate this: `save_model_weights_chunked` / `load_model_weights_from_shm` / `load_model_weights_chunk` / `extract_model_weights` / `load_weights_from_refs`.

### B5. Dropped-params diagnostics for weight synchronisation

**File**: `verl/workers/rollout/vllm_rollout/utils.py`

In the non-FP8 weight-sync branch, we collect the set of loaded parameter names returned by each sub-model's `load_weights()`, then diff it against the sent_names after applying the `hf_to_vllm_mapper` prefix mapping and folding `stacked_params_mapping`; parameter names not consumed on the vLLM side get a WARNING on rank 0. The fold rule uses `.qkv` (not `.qkv_proj`) for `visual.*`, and `.qkv_proj / .gate_up_proj` for `language_model.*`, cutting down false positives in the HunyuanOCR scenario.

### B6. Extra controls on the rollout sampling stage

**Involves**: `verl/workers/config/rollout.py`, `verl/workers/rollout/vllm_rollout/vllm_async_server.py`.

Three new fields on `RolloutConfig`:

- `bad_words: Optional[list[str]]`: forwarded directly to vLLM's `SamplingParams.bad_words`;
- `add_vision_logit_bias: bool = False`: when enabled, six token ids (`image_start / image_token / image_end / video_start / video_token / video_end`) are read off the processor and given a `-1e9` logit bias before vLLM samples;
- `vision_logit_bias_value: float = -1e9`.

### B7. NCCL / GLOO network variable forwarding to Ray workers

**Involves**: `verl/trainer/constants_ppo.py`, `verl/utils/distributed.py`.

Added a whitelist `_NCCL_NETWORK_FORWARD_KEYS`, organised into 7 subsystem groups totalling 25 environment variables: Transport enable/disable toggles, Socket interface selection, InfiniBand tuning, GPUDirect / topology, Algorithm / collective tuning, Debug / profiling, UCX (examples: `NCCL_IB_DISABLE` / `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` / `UCX_NET_DEVICES`). Only keys the driver shell has explicitly exported get forwarded into `runtime_env.env_vars`; the same key/value pairs are also packed into a sentinel var `_VERL_NCCL_FORWARDED_ENV="k1=v1,k2=v2,..."`. Workers unpack this sentinel back into `os.environ` before `torch.distributed.init_process_group()`. If the driver did not export a key, nothing is set and no default is provided.

---

## C. Mandatory hyperparameter choices

The switches below are frozen at their optimal combination in `train_grpo.sh`; understand their constraints before changing them.

### C1. `use_remove_padding=False` (do not flip to True)

The current transformers version that HunyuanOCR uses does not yet natively support the high-performance forward path for sequence packing / remove padding; flipping to `True` triggers forward errors or wrong results. Native support is on the roadmap and this note will be updated when it lands.

### C2. `use_dynamic_bsz=True` (recommended)

Packs micro-batches dynamically by the real token count on each card instead of a fixed sample-count split:

- Memory usage stays smooth during long-sequence training; individual long samples do not trigger OOM;
- At the same memory budget the effective token throughput per step is higher, so training is faster;
- Aligns better with the fixed padding length when `remove_padding` is off, cutting wasted compute.

### C3. `use_fast_processor=False` (do not flip to True)

The `HunYuanVLImageProcessorFast` (torchvision backend) in the transformers version HunyuanOCR uses still has known issues; the fix in [PR #47499](https://github.com/huggingface/transformers/pull/47499) is merged but has to wait for a transformers release before it is safe to enable. Stay on the slow (PIL) backend for now.

### C4. `bad_words` together with `add_vision_logit_bias`

`bad_words` explicitly lists tokens that must never appear in the generated sequence; they are masked out during sampling. For HunyuanOCR, the training script adds the six vision control tokens by default:

```
<｜hy_image▁pad｜> <｜hy_image▁start｜> <｜hy_image▁end｜>
<｜hy_video▁pad｜> <｜hy_video▁start｜> <｜hy_video▁end｜>
```

If any of these tokens leaks into the generation, it breaks downstream reward scoring and the actor forward's image position / attention branch handling. Turning on `add_vision_logit_bias=True` places a `-1e9` logit bias on these six tokens before sampling; combined with `bad_words` (post-mask), this provides double protection.

---

## File Manifest

We modified 20 files on the verl side and added 1 new file (`verl/utils/logger/rl_logging_board_writer.py`), totalling roughly +1413 / -59 lines. The complete diff is browsable in the fork:

```
verl/experimental/agent_loop/agent_loop.py            (A1)
verl/experimental/fully_async_policy/fully_async_main.py  (B1)
verl/experimental/one_step_off_policy/main_ppo.py     (B1)
verl/trainer/config/model/hf_model.yaml               (B1)
verl/trainer/constants_ppo.py                         (B7)
verl/trainer/main_ppo_v0.py                           (B1)
verl/trainer/ppo/ray_trainer.py                       (B2)
verl/trainer/ppo/v1/trainer_base.py                   (B1 / B2)
verl/utils/dataset/rl_dataset.py                      (B3)
verl/utils/distributed.py                             (B7)
verl/utils/logger/rl_logging_board_writer.py          (B2, new)
verl/utils/model.py                                   (A3)
verl/utils/tokenizer/tokenizer.py                     (A2)
verl/workers/config/model.py                          (B1)
verl/workers/config/rollout.py                        (B6)
verl/workers/engine/fsdp/transformer_impl.py          (A4)
verl/workers/rollout/llm_server.py                    (B4)
verl/workers/rollout/utils.py                         (A5)
verl/workers/rollout/vllm_rollout/utils.py            (B4 / B5)
verl/workers/rollout/vllm_rollout/vllm_async_server.py (A6 / B1 / B4 / B6)
```
