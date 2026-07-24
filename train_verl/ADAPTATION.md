# HunyuanOCR-1.5 · verl 适配说明

> 🌏 **English**: [ADAPTATION_en.md](./ADAPTATION_en.md)

本文档说明我们把 HunyuanOCR-1.5 的 GRPO 强化训练适配到 [verl](https://github.com/volcengine/verl) 框架时对 verl 做的所有改动，以及在训练脚本 `train_grpo.sh` 中必须固化的参数选型。

上游基线：verl 官方 `main` 分支，commit [`2b47a68`](https://github.com/volcengine/verl/commit/2b47a68b66d6fa21884990a5d4445ceba59c47e5)（2026-07-24 快照）。全部改动位于个人 fork：[pspdada/verl-HYOCR · 分支 `HYOCR`](https://github.com/pspdada/verl-HYOCR/tree/HYOCR)。

按用途分为三类：

- [A. HunyuanOCR 模型的强化适配](#a-hunyuanocr-模型的强化适配)：与模型结构、processor、xdrope、权重命名强绑定的关键修改，缺一个就无法把 HunyuanOCR 跑起来。
- [B. 功能性适配](#b-功能性适配)：不特定于 HunyuanOCR 的通用能力增强，其他模型也能受益。
- [C. 必要的参数选型](#c-必要的参数选型)：训练脚本中固化的、目前不能随意改动的开关。

---

## A. HunyuanOCR 模型的强化适配

### A1. Rollout 后折叠多模态占位符

**文件**：`verl/experimental/agent_loop/agent_loop.py`

新增 `_collapse_expanded_mm_placeholders(text, processor)` 工具函数，并在 `AgentLoopWorker` 里把 `tokenizer.decode(..., skip_special_tokens=True)` 单行调用替换成 `skip_special_tokens=False` + 折叠。

原因：rollout 结束后 `input_ids` 已经被 HF processor 展开过（HunYuanVL 会把每个 `<img_start><ph><img_end>` 展开成 `<img_start><ph>*N<img_end>`，N 等于视觉 patch 数）。如果直接 `tokenizer.decode` 再喂给 processor 二次调用，HF 的 `validate_inputs` 会用"占位符 token 数 == 图片数"来校验，只有两种失败姿势：

- `skip_special_tokens=True` → 占位符全被剥掉，count = 0；
- `skip_special_tokens=False` → count = N × num_patches，远超 num_images。

折叠把每一段连续占位符还原为单个，让 processor 能凭 `image_grid_thw` 重新展开一次。

**副作用**：无。函数用 `getattr(processor, "image_start_token", None)` 等三属性做 guard，非 VLM 或不暴露这三属性的 processor 直接原样返回。

---

### A2. `HunYuanVLProcessor` 不绑定 `get_rope_index`

**文件**：`verl/utils/tokenizer/tokenizer.py`

在 `hf_processor()` 的 processor case 里新增独立分支：

```python
case "HunYuanVLProcessor":
    pass  # 保留 processor 不注入 get_rope_index
```

原因：HunYuanVL 使用 xdrope（4 路 RoPE + `xdrope_section` 切分），Qwen2VL 的 3D `(T, H, W)` 实现与 xdrope 不兼容；HF 侧的 `HunYuanVLModel` 本身也未暴露 `get_rope_index`。上层通过 `hasattr(processor, "get_rope_index")` 判断，取不到就退回 1D `torch.arange` 生成 position_ids，与 HunyuanOCR 训练实际使用的方式一致。

**副作用**：无。新增独立 case，其他 processor 走原路径不变。

---

### A3. HunyuanVL 权重键映射到 vLLM 命名

**文件**：`verl/utils/model.py`

两处改动：

1. **`convert_weight_keys` 新增 HunyuanVL 权重前缀重写**。判断条件：`if any(k.startswith("model.language_model.") or k.startswith("model.vision_tower.") for k in state_dict)`。命中后做两阶段正则重写：
   - **Stage 1（顶层前缀）**：`model.language_model.*` → `model.*`，`model.vision_tower.*` → `vit.*`，对齐 vLLM 侧 `HunYuanVLForConditionalGeneration.hf_to_vllm_mapper` 期望的布局。
   - **Stage 2（vision 内部命名）**：处理 vLLM `HunYuanVisionTransformer` 与 HF 侧命名差异，包括 `patch_embed → embeddings`、`layer_norm1/2 → input/post_attention_layernorm`、`mlp.fc1/2 → dense_h_to_4h/dense_4h_to_h`、`patch_merger.proj_conv/proj_out → perceive.proj.0/2` 等。（`q/k/v_proj → qkv` 由 vLLM 自己的 `stacked_params_mapping` 处理，不在这里做。）

2. **`_architecture_to_auto_class` / `get_hf_auto_model_class` 增加 `AutoModelForVision2Seq` fallback**。`HunYuanVLForConditionalGeneration` 只注册在 `AutoModelForVision2Seq._model_mapping` 里，加上这条 fallback 后 verl 才能识别并加载。

**副作用**：无。权重重写的检测条件明确指向 HunyuanVL 特有前缀，其他模型 state_dict 不含这两个前缀，整个块直接跳过；`AutoModelForVision2Seq` 只作为后备路径新增，不覆盖原有优先级。

---

### A4. FSDP 之后针对 HunyuanVL 的运行时 patch

**文件**：`verl/workers/engine/fsdp/transformer_impl.py`

FSDP 包装模型后额外调用 `_patch_vlm_get_image_features(module)`，做三件事：

1. **修复 `get_image_features` / `get_video_features` 的 `StopIteration`**。HF 侧这两个函数用 `next(self.vit.parameters()).dtype` 探测 vit dtype；FSDP 展平参数后 `self.vit.parameters()` 变成空迭代器，`next(...)` 直接抛异常。patch 后改用 `engine_config.model_dtype`（默认 bf16）。

2. **重写 `HunYuanVLModel.forward` 让 3D `position_ids` 与 `create_causal_mask` 兼容**。xdrope 的 position_ids shape 是 `(bs, 4, seq)`，`transformers.masking_utils.create_causal_mask` 只接受 2D。patch 在传给 `create_causal_mask` 之前取 `position_ids[:, -1, :]`（xdrope 的第 4 平面，就是标准文本位置），decoder layer 仍然收到完整 3D 张量供 `apply_rotary_pos_emb_xdrope` 使用。

3. **修 `FSDPEngineWithLMHead.forward` 里 3D position_ids 的布局**。原实现把 `(bs, 4, seq)` `.transpose(0, 1)` 成 `(4, bs, seq)`；xdrope 的 `apply_rotary_pos_emb_xdrope` 期望的是 `(bs, 4, seq)`，去掉这个 transpose。

**副作用**：无。方法级 patch 的 guard 是 `hasattr(unwrapped, "vit")` 和 `hasattr(text_model, "layers")`；forward 里 3D 布局修正只在 `if position_ids.dim() == 3` 分支执行，其他模型走 2D 分支不变。

---

### A5. Rollout 阶段折叠已展开的 image / video 占位符

**文件**：`verl/workers/rollout/utils.py`

`qwen2_5_vl_dedup_image_tokens` 的支持列表从 `Qwen2VLImageProcessor` 单个扩展为 `("Qwen2VLImageProcessor", "HunYuanVLImageProcessor")`，并对任何暴露非空 `image_token_id` 的 processor 都启用折叠。

原因：HF `processor.apply_chat_template(...)` 返回的 `input_ids` 里，单个 `<image>` 已被展开为 N 个连续 `image_token_id`（N = 视觉 patch 数）。vLLM 引擎期望的是"每张图恰好一个占位符"，它会自己再展开一次。如果两侧都展开，vLLM 里每个已展开的 token 会拿到独立的 M/XD-RoPE position id，视觉特征错位到错误的旋转位置，模型输出流畅但完全"看不见图"（HunyuanOCR-1.5 上实测到的症状）。

**副作用**：Qwen2VL 完全不变（在支持列表里）。对暴露 `image_token_id` 属性的新型 VLM processor，会**新走 dedup 分支**。这在语义上是正解：若该 VLM 也预展开占位符，dedup 是必需的；若不预展开，`is_value` 全 False，`mask` 全保留，dedup 幂等。

---

### A6. vLLM 加载 VLM 时强制走 `auto` 权重路径

**文件**：`verl/workers/rollout/vllm_rollout/vllm_async_server.py`

`vLLMHttpServer` 检测到 `hf_config` 含 `vision_config` 且 `load_format == "dummy"` 时，强制切回 `auto`。

原因：VLM 的 ViT / M-RoPE / mm_receiver 里有非 tensor 状态只在 auto 加载路径下初始化；走 dummy 会遗漏这些初始化，rollout 输出退化为 image-blind 的重复 token 串。

**副作用**：仅对含 `vision_config` 的模型触发；纯文本模型走 dummy 路径不变。

---

## B. 功能性适配

### B1. `use_fast_processor` 开关贯穿全链路

**涉及**：`verl/experimental/fully_async_policy/fully_async_main.py`、`verl/experimental/one_step_off_policy/main_ppo.py`、`verl/trainer/main_ppo_v0.py`、`verl/trainer/ppo/v1/trainer_base.py`、`verl/workers/config/model.py`、`verl/trainer/config/model/hf_model.yaml`。

`hf_processor(name_or_path, ..., use_fast=True)` 硬编码改为读 `config.data.use_fast_processor` / `HFModelConfig.use_fast_processor`（默认 True），并在 vLLM 启动时把值注入 `mm_processor_kwargs.use_fast`，保证 rollout 侧与训练侧的图像预处理后端一致。

### B2. RLLoggingBoard 训练日志 dump

**涉及**：`verl/utils/logger/rl_logging_board_writer.py`（新增）、`verl/trainer/ppo/ray_trainer.py`、`verl/trainer/ppo/v1/trainer_base.py`。

新增 `dump_batch_for_rl_logging_board(...)`：把 verl 一个训练步的 `DataProto` batch 转成 RLLoggingBoard UI 期望的 per-sample JSONL，字段包括 `prompt / response / reward / logprobs / ref_logprobs / values / advantages / kl / reward_tokens / img_path / data_source / task_type` 等。`_log_rollout_data` 从原本单一 sink 拆成"built-in `{step}.jsonl`"和"RLLoggingBoard writer"两条独立通道；`_compute_reward_colocate` 把 reward manager 返回的 `reward_extra_keys` 持久到 `batch.extra_info`，供下游 dump 使用。

### B3. filter 后数据集的磁盘缓存

**文件**：`verl/utils/dataset/rl_dataset.py`

新增三个配置项 `enable_dataset_cache` / `dataset_cache_dir` / `dataset_cache_force_rebuild`。以 tokenizer 名 + data_files + filter 参数计算 md5 fingerprint 作为 cache key；命中直接 `datasets.load_from_disk` 返回，否则完成 `.filter(...)` 后 `save_to_disk` 到 `filtered_{hash}` 目录（先写 `.tmp` 再 rename，避免中断产生半成品）。

### B4. 多副本 vLLM 冷启动加速

**涉及**：`verl/workers/rollout/llm_server.py`、`verl/workers/rollout/vllm_rollout/vllm_async_server.py`、`verl/workers/rollout/vllm_rollout/utils.py`。

多副本部署时只让 replica 0 从磁盘加载真实权重，其余副本以 `force_load_format="dummy"` 起服，然后通过 `/dev/shm` + Ray object store 以 500 MB 为块流式广播权重到 followers。新增 `save_model_weights_chunked` / `load_model_weights_from_shm` / `load_model_weights_chunk` / `extract_model_weights` / `load_weights_from_refs` 等 async 方法配合。

### B5. 权重同步 dropped-params 诊断

**文件**：`verl/workers/rollout/vllm_rollout/utils.py`

non-FP8 weight-sync 分支里，收集每个 sub-model `load_weights()` 返回的已加载参数名集合，与 sent_names 经 `hf_to_vllm_mapper` 前缀映射 + `stacked_params_mapping` 折叠后 diff，把 vLLM 侧未消费的参数名在 rank 0 打 WARNING。折叠规则对 `visual.*` 使用 `.qkv`（而非 `.qkv_proj`），对 `language_model.*` 使用 `.qkv_proj / .gate_up_proj`，减少 HunyuanOCR 场景下的误报。

### B6. Rollout 采样阶段的额外控制

**涉及**：`verl/workers/config/rollout.py`、`verl/workers/rollout/vllm_rollout/vllm_async_server.py`。

`RolloutConfig` 新增三个字段：

- `bad_words: Optional[list[str]]`：直接透传给 vLLM 的 `SamplingParams.bad_words`；
- `add_vision_logit_bias: bool = False`：开启后从 processor 抓 `image_start / image_token / image_end / video_start / video_token / video_end` 六个 token id，在 vLLM 采样前打 `-1e9` logit bias；
- `vision_logit_bias_value: float = -1e9`。

### B7. NCCL / GLOO 网络变量转发到 Ray worker

**涉及**：`verl/trainer/constants_ppo.py`、`verl/utils/distributed.py`。

新增白名单 `_NCCL_NETWORK_FORWARD_KEYS`，按 7 组子系统组织共 25 个环境变量：Transport enable/disable toggles、Socket interface selection、InfiniBand tuning、GPUDirect / topology、Algorithm / collective tuning、Debug / profiling、UCX（例如 `NCCL_IB_DISABLE` / `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` / `UCX_NET_DEVICES` 等）。仅当 driver shell 显式 export 过某个 key 时才转发到 `runtime_env.env_vars`，同时把这些键值再打包成一个标兵变量 `_VERL_NCCL_FORWARDED_ENV="k1=v1,k2=v2,..."`。worker 在 `torch.distributed.init_process_group()` 之前解包这个标兵回写到 `os.environ`。driver 没 export 就完全不动，也不设默认值。

---

## C. 必要的参数选型

以下开关在 `train_grpo.sh` 中默认已按最佳组合固化，改动前请先确认理解各项的约束。

### C1. `use_remove_padding=False`（不要改成 True）

目前 HunyuanOCR 使用的 transformers 版本尚未原生支持 sequence packing / remove padding 相关的高性能 forward，改成 `True` 会导致 forward 报错或结果异常。原生支持工作正在推进中，届时会同步更新此说明。

### C2. `use_dynamic_bsz=True`（推荐保持）

按每张卡上真实的 token 数动态打包 micro-batch，替代按样本数固定切分：

- 长序列训练时显存利用率更平稳，避免个别长样本触发 OOM；
- 同等显存下每步实际 token 吞吐更高，训练更快；
- 与 remove_padding 关闭时的固定 padding 长度更契合，减少无效计算。

### C3. `use_fast_processor=False`（不要改成 True）

目前 HunyuanOCR 使用的 transformers 版本里 `HunYuanVLImageProcessorFast`（torchvision backend）仍有已知问题；相关修复的 [PR #47499](https://github.com/huggingface/transformers/pull/47499) 已合入，但需要等 transformers 发版后才能安全开启。当前请保持 slow backend（PIL）。

### C4. `bad_words` 配合 `add_vision_logit_bias`

`bad_words` 列表明确指定不应出现在生成序列中的 token，采样阶段被 mask 掉。对于 HunyuanOCR，默认在训练脚本里将 6 个视觉控制 token 加入 `bad_words`：

```
<｜hy_image▁pad｜> <｜hy_image▁start｜> <｜hy_image▁end｜>
<｜hy_video▁pad｜> <｜hy_video▁start｜> <｜hy_video▁end｜>
```

这些 token 一旦出现在生成序列里，会破坏后续 reward 打分和 actor 前向的图像位置 / attention 分支处理。同时开启 `add_vision_logit_bias=True` 在采样前对这 6 个 token 打 `-1e9` logit bias，配合 `bad_words`（后 mask） 提供双重保障。

---

## 相关文件清单

verl 侧共修改 20 个文件、新增 1 个文件（`verl/utils/logger/rl_logging_board_writer.py`），累计约 +1413 / -59 行。完整变更可在 fork 仓库的 diff 视图查看：

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
verl/utils/logger/rl_logging_board_writer.py          (B2, 新增)
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
