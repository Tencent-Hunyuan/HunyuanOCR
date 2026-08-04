# 数据格式与打包流水线

[English Version](./data_format.md)

## 概述

训练所需的 JSONL 是**打包后**的格式：每一行包含多条原始样本拼接而成的单个序列，长度上限为 `pack_length` 个 token。这种做法通过消除 padding 浪费来最大化 GPU 利用率。

流水线：原始 JSONL → 分词 + 计数 → 打包 → 可用于训练的 JSONL。

---

## 1. 原始 OCR JSONL 格式

原始 JSONL 每行是一条训练样本。打包脚本 `normalize_sample()` 同时支持两种输入格式，并按下述优先级解析。

### 格式 A（优先）：`conv` + `img_path_sh` / `img_path_cq`

```json
{
  "img_path_sh": "/absolute/path/to/image.png",
  "conv": [
    { "question": "提取文档图片中正文的所有信息用markdown格式表示...", "answer": "# Title\n\nBody text ..." }
  ]
}
```

| 字段                | 类型         | 是否必填 | 说明|
| --------------------------- | ------------ | :------: | ----------------------------------------------------------------- |
| `img_path_sh`/`img_path_cq` | `str`        |    ✅    | 图片绝对路径，优先取 `img_path_sh`，否则取 `img_path_cq`。        |
| `conv`                | `list[dict]` |    ✅    | 取第 0 项的 `question` / `answer`。                               |

### 格式 B（兼容）：`image_path` + `conversations`

```json
{
  "image_path": ["/absolute/path/to/image.png"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n提取文档图片中正文的所有信息用markdown格式表示..."
    },
    { "from": "gpt", "value": "# Title\n\nBody text ..." }
  ]
}
```

| 字段            | 类型         | 是否必填 | 说明|
| --------------- | ------------ | :------: | ----------------------------------------------------------------------------------------------- |
| `image_path`    | `list[str]`  |    ✅    | 绝对路径列表，取第 0 张。|
| `conversations` | `list[dict]` |    ✅    | `human`/`gpt`（或 `user`/`assistant`）交替对话；`human` 值中的 `<image>` 占位符会被去除后作为 question。 |

> 只有当样本不含 `conv` 字段时才回退到格式 B。两种格式最终都会被规范化为 `image` / `question` / `answer`三元组。

## 2. 打包流水线

`tools/pipeline_count_and_pack.py` 做两件事：

1. **计数阶段**（并行）：对每条样本分词，把 token 数写入 `count_output_dir/*.jsonl`
2. **打包阶段**：按 First-Fit Decreasing 贪心策略，把样本打包到每行 `pack_length` 个 token 的上限

### 配置文件：`configs/data_list.txt`

纯文本文件，每行一个路径：

```
/path/to/dataset_A.jsonl
/path/to/dataset_B.jsonl
/path/to/dataset_C.jsonl
# 以 # 开头的行会被忽略
```

### 运行

```bash
MODEL_PATH=/path/to/HunyuanOCR \
INPUT_LIST=./configs/data_list.txt \
PACK_OUTPUT=./data/parsing_packed_20480.jsonl \
    bash scripts/pack_data.sh
```

### 可调参数（环境变量）

| 环境变量              |                                     默认值 | 含义                                                    |
| --------------------- | -----------------------------------------: | ------------------------------------------------------- |
| `MODEL_PATH`          |                                    *(必填)* | HunyuanOCR 基座模型目录（用于 tokenizer + processor）   |
| `INPUT_LIST`          |                    `./configs/data_list.txt` | 原始 JSONL 清单文件路径                                 |
| `COUNT_OUTPUT_DIR`    |                 `./data/parsing_jsonl_count` | 计数阶段输出的临时目录                                  |
| `PACK_OUTPUT`         |     `./data/parsing_packed_{PACK_LEN}.jsonl` | 最终打包好的 JSONL                                      |
| `PACK_LEN`            |                                     `20480` | 每行打包序列的最大长度                                  |
| `NUM_PROCESSES`       |                                       `32` | 计数阶段的多进程 worker 数                              |
| `THREADS_PER_PROCESS` |                                        `8` | 每个计数 worker 的线程数                                |
| `LOG_FILE`            |                            `pack_data.log` | 进度日志路径                                            |

### 输出格式（打包后的 JSONL）

每一行是一个 JSON 数组，数组中的每项是训练 dataset 实际消费的样本：

```json
[
    {
        "image": "/abs/path/1.png",
        "question": "提取图片中的文字",
        "answer": "识别出的正文",
        "num_tokens": 4123
    },
    {
        "image": "/abs/path/2.png",
        "question": "提取图片中的文字",
        "answer": "识别出的正文",
        "num_tokens": 4444
    }
]
```

数组内样本的 `num_tokens` 总和不超过 `pack_length`。`cu_seqlens` 由训练 collator 根据实际处理后的序列重新计算，不写入 JSONL。

### 性能提示

- CPU 密集：随 `NUM_PROCESSES` 近似线性扩展。128 核机器上建议 `NUM_PROCESSES=32~64`。
- 图片数据**不需要**预加载，计数阶段只做文本分词；图片在训练时按需加载。
- 首次运行会生成计数缓存；使用相同的输入清单重跑时，计数阶段会自动跳过。

## 3. 校验打包结果

对打包后的 JSONL 做健康检查：

```bash
python -c "
import json
with open('./data/parsing_packed_20480.jsonl') as f:
    for i, line in enumerate(f):
        pack = json.loads(line)
        print(f'pack {i}: {len(pack)} samples, '
              f'{sum(item.get(\"num_tokens\", 0) for item in pack)} tokens')
        if i >= 3: break
"
```

预期输出类似：

```
pack 0: 7 samples, 20438 tokens
pack 1: 5 samples, 19821 tokens
pack 2: 12 samples, 20301 tokens
pack 3: 3 samples, 18654 tokens
```

每个 pack 都接近 `pack_length=20480`，这正是打包的目标。
