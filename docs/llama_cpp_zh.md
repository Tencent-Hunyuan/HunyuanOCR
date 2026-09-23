# 使用 llama.cpp 的 PC 端部署

[English Version](./llama_cpp.md)

HunyuanOCR-1.5 可通过 [`llama.cpp`](https://github.com/ggml-org/llama.cpp) 在 **CPU / 消费级 GPU / 笔记本** 上部署：把基座模型（可选还有 DFlash 草稿）转换为 GGUF 格式，然后用 OpenAI 兼容的 `llama-server` 提供服务。

> ✅ **DFlash 已并入上游。** HunyuanOCR 的 DFlash 投机解码已经通过 [PR #28890](https://github.com/ggml-org/llama.cpp/pull/28890) 合并进 `ggml-org/llama.cpp`（提交 [`828fdf2`](https://github.com/ggml-org/llama.cpp/commit/828fdf282e195300c2965bd9511807e24ed53bdb)，首个包含它的构建为 [`b11103`](https://github.com/ggml-org/llama.cpp/releases/tag/b11103)）。此前的 DFlash fork 不再需要，基座模型与 DFlash 都直接用上游 `master`（或任意 `>= b11103` 的构建）即可。

如果你之前按旧版文档操作过，参数改名对照见 [§5 从旧 fork 迁移](#5-从旧-fork-迁移)。

---

## 1. 克隆并编译 llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# 如果有 NVIDIA GPU 并希望 CUDA 加速，追加 -DGGML_CUDA=ON
cmake -B build -DLLAMA_BUILD_EXAMPLES=ON
cmake --build ./build --config Release -j
```

确认当前代码版本已经包含 DFlash：

```bash
build/bin/llama-server --help | grep -A2 -- --spec-type
# 列出的类型里必须有 draft-dflash
```

## 2. 为权重转换准备 Python 环境

```bash
uv venv --python 3.12 venv-llamacpp
source venv-llamacpp/bin/activate
uv pip install huggingface_hub transformers torch openai
```

## 3. 下载 HunyuanOCR 权重并转换为 GGUF

```bash
hf download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

HF 模型仓库把 DFlash 草稿放在 `dflash/` 子目录里，所以上面这一条命令会把基座和草稿一起拉下来。

```bash
# 语言 / 解码器权重 → hyocr-f16.gguf
python3 convert_hf_to_gguf.py \
    --outfile ./HunyuanOCR/hyocr-f16.gguf \
    --outtype f16 \
    ./HunyuanOCR

# 视觉（mmproj）权重 → mmproj-hyocr-f16.gguf
python3 convert_hf_to_gguf.py \
    --outfile ./HunyuanOCR/mmproj-hyocr-f16.gguf \
    --outtype f16 \
    --mmproj \
    ./HunyuanOCR
```

### 3.1 转换 DFlash 草稿权重（可选）

只用基座模型可以跳过这一步。`--target-model-dir` 指向 HunyuanOCR 基座的 HF 检查点（用于 tokenizer / hidden size / 层数），位置参数指向 DFlash 检查点目录。

```bash
python3 convert_hf_to_gguf.py \
    --outfile ./HunyuanOCR/hyocr-dflash-bf16.gguf \
    --outtype bf16 \
    --target-model-dir ./HunyuanOCR \
    ./HunyuanOCR/dflash
```

---

## 4. 启动 OpenAI 兼容服务

### 4.1 仅基座模型

```bash
build/bin/llama-server \
    --model  "./HunyuanOCR/hyocr-f16.gguf" \
    --mmproj "./HunyuanOCR/mmproj-hyocr-f16.gguf" \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096 \
    -fa on --jinja
```

服务端点为 `http://<host>:8080/v1/chat/completions`，别名 `HYVL`。

### 4.2 启用 DFlash 投机解码

```bash
build/bin/llama-server \
    --model       "./HunyuanOCR/hyocr-f16.gguf" \
    --mmproj      "./HunyuanOCR/mmproj-hyocr-f16.gguf" \
    --spec-draft-model "./HunyuanOCR/hyocr-dflash-bf16.gguf" \
    --spec-type draft-dflash --spec-draft-n-max 15 \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096 \
    --parallel 1 \
    -fa on --jinja
```

DFlash 相关关键参数：

| 参数                                 | 含义                                                                                                                       |
| :----------------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| `--spec-draft-model <path>`（`-md`） | DFlash 草稿模型的 GGUF 路径                                                                                                |
| `--spec-type draft-dflash`           | 选择 DFlash 投机解码。该类型可以从草稿 GGUF 的元数据里自动识别，显式指定只是为了不产生歧义。                               |
| `--spec-draft-n-max 15`              | 每个投机步的草稿 token 数。发布的草稿模型 `block_size = 16`，因此上限是 15，超过会被自动截断；默认值只有 3，需要显式设置。 |
| `--parallel 1`                       | 单串行 slot（在 PC 上跑 DFlash 时推荐）                                                                                    |
| `-fa on`                             | 目标模型与草稿模型都启用 flash attention                                                                                   |
| `--jinja`                            | 使用随权重发布的 chat template                                                                                             |

对分辨率特别高的输入，可以额外调大 `--batch-size / --ubatch-size`（我们自己的机器上用的是 `8192`），让整块图像在一个 batch 里完成 prefill。

### 4.3 实测 DFlash 加速

上游 PR 给出的数据：Apple M5 Pro（6 性能核 + 12 能效核，48 GB 统一内存）、macOS 26.4.1、Metal 后端、Release 构建，跑本仓库自带的 26 张文档图 OCR 请求：

| 指标         | 数值                      |
| :----------- | :------------------------ |
| 草稿接受率   | 约 0.5                    |
| 平均接受长度 | 8.6                       |
| 输出         | 与非投机解码逐字节一致    |
| decode       | 84.6 → 167.1 t/s（约 2×） |
| prefill      | 232 → 192 t/s（慢约 17%） |

prefill 变慢是因为草稿模型同样要编码整个 prompt，所以短输出场景端到端基本持平，长结构化输出（稠密文档、表格、公式）加速明显。

---

## 5. 从旧 fork 迁移

| 旧（fork）             | 新（上游）                                                   |
| :--------------------- | :----------------------------------------------------------- |
| `--model-draft <path>` | `--spec-draft-model <path>`（`-md`，`--model-draft` 仍可用） |
| `--dflash`             | `--spec-type draft-dflash`                                   |
| `--draft-max 16`       | `--spec-draft-n-max 15`                                      |

草稿权重的 GGUF 转换命令没有变化。另外，旧文档里的 `--draft-max 16` 超过了训练时的 block size，实际上一直被截断为 15。

---

## 6. 快速验证

我们在 [`llama_cpp/`](../llama_cpp) 下附带一个最小的 OpenAI 兼容客户端和 26 张 OCR 测试图片，用于端到端冒烟测试。

### 6.1 安装客户端依赖

```bash
pip install openai
```

### 6.2 运行

```bash
cd llama_cpp
python chat.py
```

`chat.py` 默认连接 `http://127.0.0.1:8080/v1`、别名 `HYVL`（与上面 `llama-server` 的启动命令匹配），读取 `test_assets/data.jsonl`，把第一个 `ocr` 样本发到服务器，打印响应与每条耗时，并把所有内容 tee 到 `logs/chat_<timestamp>.log`。

在 `chat.py` 顶部可调整客户端行为：

```python
BASE_URL     = "http://127.0.0.1:8080/v1"
MODEL        = "HYVL"
MAX_REQUESTS = 10                # 总请求上限
TYPE_LIMITS  = {"ocr": 1}        # 按类型限制；设为 None 关闭
```

### 6.3 示例输出

```
=== [ocr] ocr/0.png ===
Prompt: 请提取文档图片中正文的所有信息用 markdown 格式表示。
ring, and Jacobson semisimple, by Corollary 8.35(ii)]. The factor module
$ J^{q}/J^{q+1} $ is an $ (R/J) $-module; hence, by Corollary 8.43,
$ J^{q}/J^{q+1} $ is a semisimple module, and so it can be decomposed into a
direct sum of (possibly infinitely many) simple $ (R/J) $-modules. ...
...
[elapsed] 6.885s

[total] 1 items (ocr=1), elapsed: 6.885s
```

看到响应被流式返回，末尾出现 `[total]` 行，说明 llama.cpp 部署（含 / 不含 DFlash）工作正常。

启用 DFlash 时，server 日志还会输出草稿接受率，这是确认投机解码确实在生效的最快方式。
