# HunyuanOCR-1.5 · verl RL 训练配套 <!-- omit in toc -->

> 🌏 **English**: [README_en.md](./README_en.md)

本目录（`train_verl/`）提供 HunyuanOCR-1.5 的强化学习训练全套配套：GRPO 训练脚本、reward 打分系统、数据准备与 Ray 集群启动脚本，以及针对 HunyuanOCR-1.5 适配过的 [verl](https://github.com/volcengine/verl) fork。本目录的目标是**让 HunyuanOCR-1.5 的 RL 训练最小可运行**，不是对技术报告实验的完整复现；仅提供跑通训练所必需的脚手架与关键适配。

底层 RL 框架采用 verl（FSDP2 actor + vLLM 异步 rollout）。上游基线为 verl 官方 `main` 分支 commit [`2b47a68`](https://github.com/volcengine/verl/commit/2b47a68b66d6fa21884990a5d4445ceba59c47e5)（2026-07-24 快照）；HunyuanOCR-1.5 适配所需的所有改动位于 fork [`pspdada/verl-HYOCR` · 分支 `HYOCR`](https://github.com/pspdada/verl-HYOCR/tree/HYOCR)，详细说明见 [ADAPTATION.md](./ADAPTATION.md)（[English](./ADAPTATION_en.md)）。

---

## 目录 <!-- omit in toc -->

- [1. 目录结构](#1-目录结构)
- [2. 环境准备](#2-环境准备)
- [3. Judge server 部署](#3-judge-server-部署)
- [4. Ray 集群](#4-ray-集群)
- [5. 启动训练](#5-启动训练)
- [6. Reward 系统](#6-reward-系统)
- [7. Checkpoint 与导出](#7-checkpoint-与导出)
- [8. RLLoggingBoard（可选）](#8-rlloggingboard可选)
- [9. 常见坑](#9-常见坑)
- [10. 相关文档](#10-相关文档)
- [11. 引用](#11-引用)

---

## 1. 目录结构

`train_verl/` 顶层由四部分构成：训练脚本 + 环境安装、数据准备、reward 打分系统、集群与 checkpoint 辅助工具。

```
train_verl/
├── train_grpo.sh          # 主入口：GRPO 训练启动脚本
├── install_cu13.sh        # CUDA 13 环境一键安装脚本（pip 部分）
├── reward_ocr.py          # verl reward function 入口（compute_score）
├── ADAPTATION.md          # verl 适配说明
├── data/                  # 原始 JSONL → verl parquet
├── reward/                # Reward 打分系统（分发器 + 每任务 rule scorer + judge 客户端）
└── utils/                 # Ray 集群启动 + FSDP checkpoint 合并等辅助脚本
```

`verl-HYOCR` fork 单独一个仓库，克隆到本地 `verl/` 目录并 `pip install -e .` 后本目录代码即可使用（见 [第 2 节](#2-环境准备)）。

<details>
<summary>展开完整目录树</summary>

```
train_verl/
├── train_grpo.sh                       # 主入口：GRPO 训练启动脚本
├── install_cu13.sh                     # CUDA 13 环境一键安装脚本（pip 部分）
├── reward_ocr.py                       # verl reward function 入口（compute_score）
├── ADAPTATION.md                       # verl 适配说明
│
├── data/
│   └── prepare_data.py                 # 原始 JSONL → verl parquet
│
├── reward/                             # Reward 打分系统
│   ├── ocr_scorer.py                   # 分发器 + judge 模型客户端
│   ├── ocr_utils.py                    # TaskType 枚举 + 通用 helper
│   ├── task_scorers/                   # 每个任务的规则打分模块
│   │   ├── spotting.py                 # 检测识别
│   │   ├── layout.py                   # 版面分析
│   │   ├── parsing.py                  # 通用解析
│   │   ├── ie/ie_eval.py               # 信息抽取（JSON 字段级评分 + parsing fallback）
│   │   └── chart_deplot/               # 图表解析（csv / mermaid / md-list）
│   └── utils/                          # 通用 helper（prompt / request / server / judge_server_routes.json）
│
├── utils/
│   ├── ckpt/
│   │   └── merge_fsdp_ckpt_to_hf.sh    # FSDP2 shard → HuggingFace 标准权重（可选）
│   └── ray/
│       ├── start_ray.sh                # 拉起 Ray 集群（单机 / 多节点）
│       ├── stop_ray.sh
│       └── network_envs.sh             # NCCL / IB / GLOO 网络环境变量模板
```

</details>

---

## 2. 环境准备

### 2.1 硬件建议 <!-- omit in toc -->

- GPU：NVIDIA Hopper 或 Ampere 架构（H20 / H800 / A100 等），单机 8 卡起步，多节点通过 InfiniBand 互联。
- 显存：单卡至少 80 GB（HunyuanOCR-1.5 在 FSDP2 + vLLM async rollout 组合下按 token 动态打包 micro-batch）。
- 磁盘：训练 checkpoint 每步产出较大，建议输出目录挂在高吞吐 NVMe 或分布式共享存储。

### 2.2 宿主机前置 <!-- omit in toc -->

在开始装 Python 包之前，宿主机（或 docker 镜像）需已具备：

- NVIDIA Driver ≥ 535.161.08
- CUDA ≥ 13.0（推荐 13.3）
- Python 3.12.11（下面会用 conda 创建 env）
- 系统级 cuDNN、NCCL、nsight、ffmpeg 等由宿主机 / 镜像层提供

### 2.3 创建 conda 环境 <!-- omit in toc -->

```bash
conda create -n verl-hyocr python==3.12.11 -y
conda activate verl-hyocr
```

后续所有命令都在 `verl-hyocr` 这个 env 内执行。

### 2.4 安装依赖 <!-- omit in toc -->

```bash
cd train_verl
USE_MEGATRON=0 USE_SGLANG=0 bash install_cu13.sh
```

### 2.5 安装 verl-HYOCR fork <!-- omit in toc -->

```bash
git clone -b HYOCR https://github.com/pspdada/verl-HYOCR.git verl
cd verl
pip install -e . --no-deps
cd ..
```

fork 的仓库名是 `verl-HYOCR`，但训练脚本以 `python -m verl.trainer.main_ppo` 方式导入 `verl.xxx` 模块，因此本地目录必须命名为 `verl`；`git clone <url> verl` 会直接把 clone 目标目录名指定为 `verl`。这样 ADAPTATION.md 里列出的 `verl/experimental/...` 等路径也能与本地目录一一对应。

### 2.6 准备数据 <!-- omit in toc -->

`data/prepare_data.py` 把原始 JSONL 转成 verl parquet。每行 JSONL 需要的字段如下。

必需字段：

- `image` 或 `images`：图像路径（单张为 str，多图为 list）
- `ref_answer`：参考答案
- `prompt`：文本 prompt，或完整的 chat messages list。缺省时脚本会在单条 user turn 前面按图数量拼 `<image>` 占位符。
- `question`：rollout 时的用户问题；vqa / translation 会拼进 judge 提示词，rule-based 任务写空串亦可，字段必须存在。
- `task_type`：驱动 reward 分发的关键字段，取值：`spotting` / `layout` / `parsing` / `ie` / `chart_deplot` / `translation` / `vqa`。若整个文件是同一任务，可以在源 JSONL 里省略这个字段，用命令行 `--task-type` 统一打标。

可选字段：

- `source_lang_text` / `target_lang`：translation 专属。

输出 parquet schema（每行）：

```python
{
    "data_source":  str,        # reward-router key，默认取 task_type
    "prompt":       list[dict], # chat messages，content 是含 <image> 占位符的字符串
    "images":       list[str],  # 绝对路径列表，数量与 <image> 数量对齐
    "reward_model": {"style": "rule", "ground_truth": str},
    "extra_info": {
        "task_type":        str,        # 必填；驱动 reward 分发
        "question":         str,        # 可选；vqa / translation 用
        "img_path":         list[str],  # 可选；供 RLLoggingBoard 展示
        "source_lang_text": str,        # 可选；translation 用
        "target_lang":      str,        # 可选；translation 用
    },
}
```

典型调用：

```bash
python data/prepare_data.py \
    --inputs /path/to/a.jsonl /path/to/b.jsonl \
    --output_train /path/to/train.parquet \
    --output_val   /path/to/val.parquet \
    --val_ratio 0.02
```

更多参数直接跑 `python data/prepare_data.py -h` 查看。

---

## 3. Judge server 部署

Translation 与 VQA 两类任务通过 **judge 模型**打分；其余五类走本地规则打分，无需 judge。因此若你的训练数据里包含 translation / vqa 样本，需要额外把 judge server 起起来。

用 vLLM 起 OpenAI 兼容 server（以 Qwen/Qwen3-30B-A3B 为例）：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 --max-model-len 16384
```

server 起起来后把地址填进 `reward/utils/judge_server_routes.json`：

```json
{
  "Qwen/Qwen3-30B-A3B": ["10.0.0.1:8000", "10.0.0.2:8000"]
}
```

支持多副本，reward 侧会做 round-robin 负载均衡。如果不打算用其他 judge 模型，`train_grpo.sh` 里的 `RM_SYSTEM_JUDGE_MODEL_NAME` 保持默认即可；换其他模型则同步改 `RM_SYSTEM_JUDGE_MODEL_NAME` 和 `judge_server_routes.json` 的 key。

---

## 4. Ray 集群

verl 训练依赖一个提前起好的 Ray 集群（driver 通过 `RAY_ADDRESS=auto` 附着上去）。集群和训练进程解耦的好处是 dashboard、slave 节点可以跨越 driver 重启保持存活。

### 4.1 单机 <!-- omit in toc -->

```bash
bash utils/ray/start_ray.sh local
```

### 4.2 多节点 <!-- omit in toc -->

```bash
# head 节点（INDEX=0）
NNODES=<N> INDEX=0 bash utils/ray/start_ray.sh <N> 0

# 其余节点
NNODES=<N> INDEX=<i> bash utils/ray/start_ray.sh <N> <i>
```

`start_ray.sh` 内部会 `source utils/ray/network_envs.sh`，把 NCCL / IB / GLOO 相关环境变量按你集群实际情况 export。多节点场景下通常需要根据 `ibdev2netdev` 和 `ip -o link show` 的输出调整以下项：

- `NCCL_IB_HCA`
- `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`
- `UCX_NET_DEVICES`

如果 IB 或 GDR 有兼容性问题，可以逐级降级到 A→B→C 三档（IB+GDR → IB 无 GDR → 纯 TCP，`DISABLE_IB=1`），`network_envs.sh` 顶部注释里有说明。

停机：

```bash
bash utils/ray/stop_ray.sh
```

---

## 5. 启动训练

一句话启动：

```bash
bash train_grpo.sh
```

`train_grpo.sh` 会自动完成 Ray 集群 attach、reward function 注册、FSDP2 actor + vLLM rollout 配置、以及运行时目录初始化。

### 5.1 关键环境变量 <!-- omit in toc -->

顶层几项是最常调整的，通过 env 覆盖：

| 变量                         | 默认                        | 含义                                                 |
| :--------------------------- | :-------------------------- | :--------------------------------------------------- |
| `MODEL_PATH`                 | `/path/to/HunyuanOCR/model` | HunyuanOCR-1.5 权重目录（HF 格式）                   |
| `TRAIN_FILES`                | `/path/to/train.parquet`    | 训练 parquet                                         |
| `VAL_FILES`                  | `/path/to/val.parquet`      | 验证 parquet                                         |
| `OUTPUT_DIR`                 | `./outputs`                 | 训练产物根目录                                       |
| `NNODES`                     | `2`                         | 节点数                                               |
| `NGPUS_PER_NODE`             | `8`                         | 每节点 GPU 数                                        |
| `PROJECT_NAME`               | `hyocr_1_5_verl`            | 项目名（用于组织输出子目录）                         |
| `EXPERIMENT_NAME`            | `hyocr_1_5_grpo`            | 实验名                                               |
| `REWARD_FN_PATH`             | `./reward_ocr.py`           | reward 函数文件路径                                  |
| `RM_SYSTEM_JUDGE_MODEL_NAME` | `Qwen/Qwen3-30B-A3B`        | Judge 模型名，需匹配 `judge_server_routes.json` 的 key |

单机 8 卡示例：

```bash
NNODES=1 NGPUS_PER_NODE=8 \
MODEL_PATH=/data/hyocr_1_5 \
TRAIN_FILES=/data/train.parquet \
VAL_FILES=/data/val.parquet \
OUTPUT_DIR=/data/outputs \
bash train_grpo.sh
```

多节点示例：先在所有节点起 Ray，再在 head 节点跑：

```bash
NNODES=4 NGPUS_PER_NODE=8 \
MODEL_PATH=/data/hyocr_1_5 \
TRAIN_FILES=/data/train.parquet \
VAL_FILES=/data/val.parquet \
OUTPUT_DIR=/data/outputs \
bash train_grpo.sh
```

其余细粒度开关（batch size、KL 系数、rollout 采样参数、logging 频率等）在 `train_grpo.sh` 里都有行内注释，需要调时直接编辑脚本即可，不必所有开关都往 env 里搬。

### 5.2 运行产物 <!-- omit in toc -->

训练开始后，`OUTPUT_DIR/${PROJECT_NAME}/${EXPERIMENT_NAME}/` 下会形成如下结构：

```
├── ckpt/                       # verl 保存的 FSDP2 sharded checkpoint
├── tensorboard/                # TensorBoard 事件文件
├── metrics.jsonl               # 每步指标（VERL_FILE_LOGGER_PATH）
├── stdout.log                  # 训练标准输出
└── rl_logging_board/           # RLLoggingBoard per-sample JSONL dump（见第 8 节）
```

---

## 6. Reward 系统

### 6.1 数据流 <!-- omit in toc -->

verl 侧的 `NaiveRewardManager` 在每次 rollout 结束后调用：

```
compute_score(data_source, solution_str, ground_truth, extra_info)
    → OCRScorer.process_scoring({...})
        → task_scorers/<xxx>.py 里的 process_<task>_task(response, ref_answer)
```

- `reward_ocr.py`：verl 侧 reward function 入口，负责 kwargs 到内部 scoring_obj 的映射。
- `reward/ocr_scorer.py`：`OCRScorer.process_scoring` 是分发器，按 `task_type` 走七条分支。
- `reward/task_scorers/`：每个任务一个入口函数 `process_<task>_task(response, ref_answer) -> dict`。

### 6.2 七种任务的打分策略 <!-- omit in toc -->

| Task         | 分发条件（`TaskType.is_xxx`）   | 打分方式                                                                          |
| :----------- | :------------------------------ | :-------------------------------------------------------------------------------- |
| spotting     | `"spotting"` / `"检测识别"`     | 规则：格式探测 + IoU 匹配 + 归一化编辑距离                                        |
| layout       | `"layout"` / `"版面分析"`       | 规则：hy-meta / JSON 双格式，F1 模式 or 编辑距离模式                              |
| parsing      | `"parsing"` / `"通用解析"`      | 规则：字符准确率 + TEDS 表格评分                                                  |
| ie           | `"ie"` / `"信息抽取"`           | 规则：JSON 字段级 exact_match + edit-distance；ref 非 JSON 时 fallback 到 parsing |
| chart_deplot | `"chart_deplot"` / `"图表解析"` | 规则：按 ref 格式分发到 csv / tree / flowchart 评测                               |
| translation  | `"translation"` / `"翻译"`      | Judge 模型：0-5 分再按分段线性映射到 [0, 1]                                       |
| vqa          | `"vqa"` / `"视觉问答"`          | Judge 模型：Judgement 0 / 1                                                       |

判定采用 substring 包含（英文别名 case-insensitive 匹配 + 中文别名字面量匹配），只要 task_type 包含对应关键词就会命中相应分支。

### 6.3 自定义新任务 <!-- omit in toc -->

想加一个新的 rule-based 任务时按三步走：

1. `reward/ocr_utils.py` 里给 `TaskType` 加一个英文枚举成员 + 中文别名成员（例如 `MY_TASK = "MyTask"` / `MY_TASK_ZH = "我的任务"`），再加一个 `is_my_task` 分类方法（一行 dispatch 到 `_matches`）。
2. `reward/task_scorers/` 下新建 `my_task.py`，实现 `process_my_task_task(response, ref_answer) -> dict`（返回 `{"analysis", "is_valid", "reward"}` 三个键即可）；内部建议加一层 try/except 包成 `reward=-1.0`，保证不 raise 到分发器。
3. `reward/ocr_scorer.py` 里 `process_scoring` 加一段三行的分发分支，`reward/task_scorers/__init__.py` 里 re-export 新入口。

Judge-model 类型的新任务多一步：往 `reward/utils/prompt.py` 的 `JUDGE_PROMPTS` 里加 system / template 两段，`validate_format_response` 里加对应的抽分正则。

---

## 7. Checkpoint 与导出

### 7.1 训练产物 <!-- omit in toc -->

verl 保存的 checkpoint 位于 `OUTPUT_DIR/${PROJECT_NAME}/${EXPERIMENT_NAME}/ckpt/global_step_${N}/`：

```
global_step_${N}/actor/
├── model_world_size_${WS}_rank_*.pt   # FSDP2 每 rank 的 shard
├── optim_world_size_${WS}_rank_*.pt
├── extra_state_world_size_${WS}_rank_*.pt
├── fsdp_config.json
└── huggingface/                        # config / tokenizer / processor / chat_template
```

### 7.2 合并成 HuggingFace 权重（可选） <!-- omit in toc -->

若需要把 FSDP2 sharded checkpoint 合并成标准 HF safetensors（例如用于离线评测或发布），运行：

```bash
# 合并单个 step
STEPS=100 bash utils/ckpt/merge_fsdp_ckpt_to_hf.sh

# 批量
STEPS="25 50 75 100" bash utils/ckpt/merge_fsdp_ckpt_to_hf.sh

# 合并 CKPT_ROOT 下所有已存在的 step
STEPS=all bash utils/ckpt/merge_fsdp_ckpt_to_hf.sh
```

底层调用 `python -m verl.model_merger merge --backend fsdp ...`，与训练时 `actor.checkpoint.save_contents` 包含 `'hf_model'` 得到的结果完全一致。默认原地写入 `actor/huggingface/`。

---

## 8. RLLoggingBoard（可选）

`train_grpo.sh` 打开了 verl 的 `_log_rollout_data` 双通道 sink：内置 `{step}.jsonl` 之外，还额外把每个训练步的 rollout 数据 dump 成 [RLLoggingBoard](https://github.com/HarderThenHarder/RLLoggingBoard) UI 期望的 per-sample JSONL（含 prompt / response / reward / logprobs / advantages / kl / img_path / data_source / task_type 等字段），落在 `OUTPUT_DIR/${PROJECT_NAME}/${EXPERIMENT_NAME}/rl_logging_board/`。

安装 RLLoggingBoard 后指向这个目录即可打开可视化面板，逐样本查看 rollout 质量、reward 分布、KL 曲线等。

---

## 9. 常见坑

- **`use_fast_processor=False` 请保持默认**：当前 transformers 版本的 `HunYuanVLImageProcessorFast`（torchvision backend）仍有已知问题，相关修复 [PR #47499](https://github.com/huggingface/transformers/pull/47499) 已合入但需等发版。目前保持 slow backend（PIL）。
- **`use_remove_padding=False` 请保持默认**：当前 transformers 版本尚未原生支持 sequence packing / remove padding 相关的高性能 forward，改成 True 会导致 forward 报错或结果异常。
- **`bad_words` + `add_vision_logit_bias` 是双重保险**：训练脚本默认把 6 个视觉控制 token 加进 `bad_words`，同时开启 `add_vision_logit_bias=True` 在采样前打 `-1e9` logit bias。生成序列出现这些 token 会破坏 reward 打分与 actor 前向的图像分支处理，两者建议一起开。
- **单机也能跑，多机需要 export NCCL 网络变量**：`network_envs.sh` 只 export driver 已经设置过的 NCCL / IB / GLOO 变量，driver 未 export 就完全不动。单机默认走 NVLink / 本地 socket 即可；多机需要根据 `ibdev2netdev` 输出配置 `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` 等，具体可以参照 [ADAPTATION.md B7](./ADAPTATION.md#b7-nccl--gloo-网络变量转发到-ray-worker) 里 25 个白名单变量。

---

## 10. 相关文档

- [ADAPTATION.md](./ADAPTATION.md)：verl 适配说明，按 A（HunyuanOCR 强化适配）/ B（通用功能性适配）/ C（关键参数选型）三类分组。
- [verl-HYOCR fork](https://github.com/pspdada/verl-HYOCR/tree/HYOCR)：fork 分支的完整代码。
- [verl 上游](https://github.com/volcengine/verl)：上游主库，可与 fork 做 diff 对照。
- [RLLoggingBoard](https://github.com/HarderThenHarder/RLLoggingBoard)：训练日志可视化 UI。

---

## 11. 引用

```bibtex
@article{HunyuanOCR_1_5_2026,
  title   = {{HunyuanOCR-1.5}: Making Lightweight {OCR} {VLMs} Faster and Better},
  author  = {Li, Gengluo and Wan, Xingyu and Peng, Shangpin and Wang, Weinong and Feng, Hao and Du, Yongkun and Wu, Binghong and Ruan, Zheng and Lu, Zhiqiong and Wu, Liang and Lyu, Pengyuan and Shen, Huawen and Lin, Zibin and Hu, Shijing and Yang, Jieneng and Wen, Hongbing and Yu, Guanghua and Liu, Hong and Wang, Bochao and Ma, Can and Hu, Han and Zhang, Chengquan and Zhou, Yu},
  journal = {arXiv preprint arXiv:2607.04884},
  year    = {2026}
}
```
