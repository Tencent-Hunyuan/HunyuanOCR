# 环境 B · HunyuanOCR-1.5 with vLLM nightly（CUDA 13，AR + DFlash）

[English Version](./DFlash.md)

vLLM **nightly** 环境同时支持**自回归（AR）**解码和 **DFlash 投机解码**（无损加速，输出越长收益越大）。DFlash 草稿模型位于 HF 模型仓库的 `dflash/` 子目录下，与基座模型一起下载即可（见 §2）。这套环境比环境 A 复杂：需要 CUDA 13（torch cu130）以及一个 CUDA 13 compat 库。若需要原生 transformers 推理，请用 [`transformers`](./transformers_zh.md)（本环境里的 transformers 5.5.3 不支持它）。

> 已在 Python 3.12 + vLLM nightly（cu130）+ NVIDIA H20（驱动 535 + 自建 CUDA 13 compat 库）上从干净环境验证，包含在 OmniDocBench 上的完整跑分（AR 与 DFlash 各 1651 张图）。§1 的三个 ⚠️ 步骤正是那次踩过的坑，**不要跳过**。

**AR 还是 DFlash？**

- 输出一致（DFlash 是无损加速），采样一致，客户端也完全通用。
- DFlash 加速随输出长度增长而变大（长文档解析显著，短输出场景相当），并且并发越低加速越明显。
- 追求最快的长文档吞吐 → DFlash；追求最简单的基线 → AR。切换只需换两个 serve 脚本。

---

## 目录

- [1. 环境安装（重要，比环境 A 复杂，按步骤执行）](#1-环境安装)
- [2. 下载权重 + 草稿模型](#2-下载权重--草稿模型)
- [3. 启动服务（单卡）](#3-启动服务单卡)
- [4. 推理](#4-推理)
- [5. 任务类型](#5-任务类型)
- [6. 文件](#6-文件)

---

## 1. 环境安装

要求：Python 3.12、NVIDIA GPU（显存 ≥ 24 GB）。**DFlash 投机解码方法只在 vLLM nightly（cu130）里注册**，0.18.1 / 0.24.0 等正式版都不支持，所以必须用 nightly。

> ⚠️ **本节有三个必做步骤（§1.1 里的两个，加上 §1.2 的 compat 库）。任何一个漏了，服务都起不来。** 这些坑来自于 nightly 版本每天滚动依赖，下面每一步都解释了**为什么**要做，别跳过。

### 1.1 安装 vLLM nightly（cu130）

走公司代理时先设置：`export http_proxy=http://<proxy>:<port> https_proxy=http://<proxy>:<port>`

```bash
pip install -U uv
uv venv --python 3.12 && source .venv/bin/activate

# Step 1：安装 vLLM nightly（cu130），会拉进 torch cu130 / transformers / flashinfer / torchcodec
uv pip install -U vllm --torch-backend=cu130 --extra-index-url https://wheels.vllm.ai/nightly
uv pip install openai pillow
```

#### ⚠️ Step 2（必做）：把 transformers 锁到 5.5.3

这是最大的坑。安装 nightly **默认拉进 transformers 5.13.0**，但 **5.13.0 会导致 vLLM 加载 HunYuanVL 崩溃**：

```
AttributeError: 'str' object has no attribute '__module__'
  （vLLM 的 hunyuan_vl_image.py 用字符串参数调用 AutoImageProcessor.register("...", ...)，
   而 transformers 5.13 改了 register() 签名，会对第一个参数取 .__module__。）
```

`vllm serve` 直接起不来，且 traceback 未必指向 transformers，很难排查。**显式降级到 5.5.3**（`--no-deps` 保留 torch / vLLM 不动）：

```bash
uv pip install "transformers==5.5.3" --no-deps
```

> 验证：`python -c "import transformers; print(transformers.__version__)"` 应打印 `5.5.3`。若你需要原生 HuggingFace transformers 推理（需要 transformers 5.13.0），那是一个**独立的、不共存的环境**，见 [`transformers`](./transformers_zh.md)，不能与本 vLLM 环境共用。

#### ⚠️ Step 3（必做）：卸载 torchcodec

nightly 还会拉进 `torchcodec`（视频解码）。它在 import 时会加载底层 FFmpeg 库（`libavutil.so.56`……），缺库会抛 `OSError`（不是 `ImportError`），vLLM 的 try/except 抓不住，直接把 `vllm serve` 打死：

```
OSError: libavutil.so.56: cannot open shared object file
  → Could not load .../torchcodec/libtorchcodec_core4.so
```

HunyuanOCR 是**图片模型**，不需要视频解码。直接卸载它，让 vLLM 走 ImportError 回退分支：

```bash
uv pip uninstall torchcodec
```

> 已验证版本：`vllm 0.23.1rc1.dev825~dev869`（后缀每天滚动）、`torch 2.11.0+cu130`、**`transformers 5.5.3`**、`flashinfer 0.6.13`、torchcodec 已卸载。cu129 的 nightly 不行，cu130 可以。

### 1.2 CUDA 13 compat 库（当宿主驱动 < 580 时必做）⚠️

`torch cu130` 需要 **≥ 580** 版本的 CUDA 用户态驱动库（`libcuda.so`）。如果宿主驱动较老（如 `nvidia-smi` 显示 535.x / CUDA 12.8），启动会报：

```
The NVIDIA driver on your system is too old (found version 12080)...
```

解决：装 `cuda-compat-13-0`，把它的 `libcuda.so.580.x` 通过 `LD_LIBRARY_PATH` 前置（不动宿主驱动，也不需要 root 权限）。若宿主驱动已 ≥ 580，跳过本节。

#### Step 1：下载 rpm

从 **NVIDIA CUDA 官方源**（RHEL 8 / x86_64）下载。先看有哪些版本：

```bash
# 列出所有可用的 cuda-compat-13-0 版本（NVIDIA 会裁剪旧版本，任何 580.x 都满足 ">=580"）
curl -sSL "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/" \
    | grep -oE 'cuda-compat-13-0-[0-9.]+-1\.el8\.x86_64\.rpm' | sort -u
```

选一个（示例用 `580.65.06`，上面列出的任一版本都行）：

```bash
BASE="https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64"
RPM="cuda-compat-13-0-580.65.06-1.el8.x86_64.rpm"
mkdir -p cuda_compat_13 && curl -sSL "$BASE/$RPM" -o "cuda_compat_13/$RPM"
```

> 其它系统换 repo 路径即可：`rhel9`、`ubuntu2204/x86_64`（.deb）等。详见 `https://developer.download.nvidia.com/compute/cuda/repos/`。

#### Step 2：解压 rpm

**方案 A（有 rpm2cpio / cpio 时）：**

```bash
cd cuda_compat_13 && rpm2cpio cuda-compat-13-0-*.x86_64.rpm | cpio -idmv && cd ..
# 解出的库位于 cuda_compat_13/usr/local/cuda-13.0/compat/
mkdir -p cuda_compat_13/extracted
cp -a cuda_compat_13/usr/local/cuda-13.0/compat/* cuda_compat_13/extracted/
```

**方案 B（无 rpm2cpio / cpio 时的纯 Python 解压器；较新的机器上这些工具往往没有）：**

```bash
python3 - <<'PYEOF'
import struct, os, lzma
rpm = [f for f in os.listdir("cuda_compat_13") if f.endswith(".rpm")][0]
data = open(f"cuda_compat_13/{rpm}", "rb").read()
# 跳过 96B lead + 两个 header（magic 8e ad e8），定位 payload
def hdr_end(buf, off):
    assert buf[off:off+3] == b'\x8e\xad\xe8'
    ni, ns = struct.unpack(">II", buf[off+8:off+16])
    return off + 16 + ni*16 + ns
pos = hdr_end(data, 96); pos = (pos+7) & ~7; pos = hdr_end(data, pos)
payload = data[pos:]                              # xz-compressed cpio (newc)
raw = lzma.decompress(payload) if payload[:6]==b'\xfd7zXZ\x00' else payload
os.makedirs("cuda_compat_13/extracted", exist_ok=True)
p = 0
while p < len(raw) and raw[p:p+6] == b'070701':
    g = lambda i: int(raw[p+6+i*8:p+14+i*8], 16)
    nsz, fsz, mode = g(11), g(6), g(1)
    name = raw[p+110:p+110+nsz-1].decode("utf-8","replace")
    doff = (p+110+nsz+3) & ~3
    fdata = raw[doff:doff+fsz]
    p = (doff+fsz+3) & ~3
    if name == "TRAILER!!!": break
    base = os.path.basename(name)
    if "compat" in name and ".so" in base:
        dst = f"cuda_compat_13/extracted/{base}"
        if (mode & 0xf000) == 0xa000:            # symlink
            if os.path.lexists(dst): os.remove(dst)
            os.symlink(fdata.decode(), dst)
        elif fsz > 0:
            open(dst, "wb").write(fdata)
print("extracted:", sorted(os.listdir("cuda_compat_13/extracted")))
PYEOF
```

任一方案，`cuda_compat_13/extracted/` 中都应包含 `libcuda.so.580.x`、`libnvidia-ptxjitcompiler.so.580.x`、`libnvidia-nvvm.so.580.x` 等，以及它们的软链接。

#### Step 3：每次启动前把它前置到 LD_LIBRARY_PATH

```bash
export LD_LIBRARY_PATH=$(pwd)/cuda_compat_13/extracted:$LD_LIBRARY_PATH
```

### 1.3 验证安装

```bash
export LD_LIBRARY_PATH=$(pwd)/cuda_compat_13/extracted:$LD_LIBRARY_PATH

# a) compat 生效：torch 能看到 GPU（不再报 "driver too old"）
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count())"
# 预期：cuda: True | gpus: 8

# b) dflash 方法已注册
python -c "from vllm.config import SpeculativeConfig; import inspect; \
print('dflash registered:', 'dflash' in inspect.getsource(SpeculativeConfig))"
# 预期：dflash registered: True

# c) transformers 是 5.5.3（不是 5.13.0）
python -c "import transformers; print('transformers:', transformers.__version__)"
# 预期：transformers: 5.5.3
```

---

## 2. 下载权重 + 草稿模型

```bash
pip install -U "huggingface_hub[cli]"
# 基座 + DFlash 草稿一次下完：草稿在 dflash/ 子目录里
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

HF 模型仓库把 DFlash 草稿（`config.json` + `dflash.py` + tokenizer + `model.safetensors`）放在 `dflash/` 子目录里，因此上面这一条命令会同时把基座和草稿都拉下来。`serve_DFlash.sh` 会默认使用 `${MODEL_PATH}/dflash` 作为 `DFLASH_PATH`；只有当你把草稿放在其它位置时才需要显式覆盖 `DFLASH_PATH`。

---

## 3. 启动服务（单卡）

> AR 与 DFlash 沿用当前统一布局中提供的服务脚本：AR 用 `inference/vLLM/serve.sh`，DFlash 用 `inference/DFlash/serve_DFlash.sh`。两者内部（vLLM 参数、采样默认值、DFlash `--speculative-config`）与旧 nightly 环境保持一致，因此下面的启动方式仍然适用。

```bash
# ⚠️ 当宿主驱动老旧时先设置 compat 库路径（见 §1.2）
export LD_LIBRARY_PATH=/ABS/PATH/cuda_compat_13/extracted:$LD_LIBRARY_PATH

# —— AR（自回归）——
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/vLLM/serve.sh

# —— DFlash（投机解码）——
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/DFlash/serve_DFlash.sh
```

就绪：AR 约 1-2 分钟；**DFlash 首次加载包含 torch.compile，约 3-5 分钟**。

```bash
curl -sf http://127.0.0.1:8000/v1/models
```

**`inference/vLLM/serve.sh`** 环境变量：`MODEL_PATH`（必填）、`GPU`、`PORT`、`GPU_MEM_UTIL`（默认 0.9）、`MAX_MODEL_LEN`、`SERVED_NAME`。
**`inference/DFlash/serve_DFlash.sh`** 接受同样一组变量，另加 `DFLASH_PATH`（默认 `${MODEL_PATH}/dflash`）、`NUM_SPEC_TOKENS`（默认 15）；它把 `GPU_MEM_UTIL` 默认设为 0.85 以给草稿模型预留空间（约 0.7 GB）。底层上，DFlash 比 AR 只多一条参数：`--speculative-config '{"method":"dflash","model":"<DFLASH_PATH>","num_speculative_tokens":15}'`。

> **多卡（8 卡满配）：** 每张卡启动一个实例（`GPU=0 PORT=8000` … `GPU=7 PORT=8007`，每次都记得设 `LD_LIBRARY_PATH`），然后跑 `python inference/vLLM/batch_infer.py --ports 8000,8001,...,8007`。

停止服务：`pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "vllm serve"`

---

## 4. 推理

> AR 与 DFlash 共用同一套客户端：`inference/vLLM/infer_vllm_client.py`（单图）和 `inference/vLLM/batch_infer.py`（批量）。它们从 `inference/utils/tasks.py` 与 `inference/utils/output_utils.py` 导入共享的任务 prompt 与输出工具（单份，无重复副本）。因此采样参数、任务 prompt、后处理三者完全一致，AR / DFlash / transformers 输出可直接对比。

### 采样参数（对齐官方设置，已内置，不需要修改）

`temperature=0.0`、`top_p=1.0`、`top_k=-1`、`repetition_penalty=1.08`，流式生成 + 尾部重复早停 + 尾部重复清洗。

### 单张图

```bash
python inference/vLLM/infer_vllm_client.py --image /path/doc.png --task-type doc_parse \
    --model tencent/HunyuanOCR --port 8000 --max-tokens 32768
```

### 批量（目录）

```bash
python inference/vLLM/batch_infer.py --image-dir /path/imgs --out-dir /path/out \
    --ports 8000 --task-type doc_parse --max-tokens 32768 --concurrency 16
```

- 每张图片输出同名 `.md`；`out-dir/results.jsonl` 记录延迟 / 字符数 / 早停 / 后处理明细。已完成条目自动跳过。

---

## 5. 任务类型

`--task-type` 选择官方推荐 prompt。列出全部：`python inference/vLLM/infer_vllm_client.py --list-tasks`

| task_type          | 说明                                                                       |
| ------------------ | -------------------------------------------------------------------------- |
| `doc_parse`        | 端到端文档解析（默认；正文 → md，表格 → HTML，公式 → LaTeX，忽略页眉页脚） |
| `structured_parse` | 结构化解析（古文 / 街景等）                                                |
| `spotting_json`    | 检测 + 识别 → JSON 数组（box 归一化 0-1000 + 文字）                        |
| `spotting_hunyuan` | 检测 + 识别 → Hunyuan 坐标格式                                             |
| `layout`           | 版面分析                                                                   |
| `layout_parse`     | 版面分析 + 全文解析                                                        |
| `chart_parse`      | 图表解析（流程图 → Mermaid，其他 → Markdown）                              |
| `formula`          | 公式解析（→ LaTeX）                                                        |
| `table`            | 表格解析（→ HTML）                                                         |
| `doc_trans_en2zh`  | 文档翻译，英 → 中                                                          |
| `trans_other2en`   | 通用场景翻译 → 英                                                          |
| `trans_other2zh`   | 通用场景翻译 → 中                                                          |

> Markdown 规整（由 `--no-doc-postprocess` 控制）**仅对 `doc_parse` 生效**。

---

## 6. 文件

```
inference/DFlash/
└── serve_DFlash.sh         # 单卡 vLLM + DFlash 启动脚本（比 AR 多一个 --speculative-config）

inference/vLLM/             # 与 AR 路径共用（详见 archive/vLLM_zh.md）
├── serve.sh                # 单卡 vLLM AR 启动脚本
├── infer_vllm_client.py    # 单图客户端   ┐ 与环境 A/C 共享；
└── batch_infer.py          # 批量推理     ┘ 输出可对比
```

> DFlash 草稿（config + dflash.py + tokenizer + model.safetensors）不提交到 Git，它来自 HF 模型仓库的 `dflash/` 子目录，与基座模型一起下载（见 §2）。

> 共享工具函数（`tasks.py`：task_type → prompt；`output_utils.py`：输出工具，含 doc_parse 规整）单份放在 `inference/utils/`，被三套环境 A/B/C 一起 import。

> `config.json` 里的 `vision_config.max_image_size` 是位置编码表形状（模型结构参数），**不要**把它当作分辨率旋钮。
