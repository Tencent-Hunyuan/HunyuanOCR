# SFT HunyuanOCR-1.5 base trên dữ liệu tiếng Việt — 1 GPU 80GB

Runbook cho fork này. Nguồn dữ liệu: `makedata/hf_upload` (1.007 train / 120
validation, 4 config: GCN, Documents, Others, Handwriting).

Tất cả script upstream vẫn giữ nguyên hành vi cũ; phần riêng của fork là
`scripts/env_single.sh`, `scripts/sft_base_1gpu.sh`, `tools/makedata_to_hyocr.py`
và cờ `--allow-empty-answer` trong `tools/pipeline_count_and_pack.py`.

---

## 0. Môi trường

```bash
conda create -n hyocr python=3.12 -y && conda activate hyocr
pip install -r requirements.txt
pip install flash-attn --no-build-isolation      # bắt buộc, xem ghi chú bên dưới
pip install binpacking lmdb
```

`train/train_hunyuan.py` gọi model với `attn_implementation="flash_attention_2"`
cứng trong code, **không có đường lùi sang eager** — thiếu `flash-attn` là chết
ngay lúc load model, không phải lúc train.

Tải model gốc:

```bash
hf download tencent/HunyuanOCR --local-dir ./HunyuanOCR
export MODEL_PATH=$PWD/HunyuanOCR
```

## 1. Chuyển dữ liệu sang định dạng raw JSONL

```bash
python tools/makedata_to_hyocr.py \
    --root /home/jaqja/New_AI/makedata/hf_upload \
    --prompt-file /home/jaqja/New_AI/makedata/ocr_prompt.md \
    --out-dir ./data/raw \
    --data-list ./data/data_list.txt
```

Sinh ra `data/raw/train.jsonl` (1.007 dòng) và `data/raw/validation.jsonl`
(120 dòng), mỗi dòng:

```json
{"img_path_sh": "/abs/000.jpg", "conv": [{"question": "<prompt>", "answer": "<markdown>"}]}
```

> **`docs/data_format.md` của upstream mô tả sai.** Doc ghi schema
> `{"image_path": [...], "conversations": [{"from","value"}]}` và packed output
> `{"packed_samples", "cu_seqlens", "total_tokens"}`. Không có dòng code nào
> trong repo đọc các key đó. `tools/pipeline_count_and_pack.py:157-184` đọc
> `img_path_sh` / `img_path_cq` + `conv[0].question` / `conv[0].answer`, và
> `train/data_processor.py:153-189` đọc `item["image"]` / `["question"]` /
> `["answer"]` với mỗi dòng packed là **một JSON array**. Converter bám theo code.

Prompt nằm ở `question`, không phải `system`: hàm pack không chuyển
`--system-prompt` vào bản ghi packed, nên system prompt được **tính token lúc
pack rồi biến mất lúc train** — đặt prompt ở đó là lệch dữ liệu.

## 2. Pack

```bash
MODEL_PATH=$MODEL_PATH \
INPUT_LIST=./data/data_list.txt \
PACK_OUTPUT=./data/packed/train_16384.jsonl \
PACK_LEN=16384 \
NUM_PROCESSES=8 THREADS_PER_PROCESS=8 \
FOREGROUND=1 \
    bash scripts/pack_data.sh
```

`ALLOW_EMPTY=1` là mặc định của fork. Upstream loại thẳng mọi mẫu có `answer`
rỗng (`if not question or not answer: return None`) — với bộ này là **25 trang
trắng** cố ý gán nhãn rỗng để dạy model im lặng. Loại chúng đi thì dạy đúng điều
ngược lại. Đặt `ALLOW_EMPTY=0` nếu muốn hành vi upstream.

Kiểm tra kết quả:

```bash
python -c "
import json
n=t=0
for line in open('./data/packed/train_16384.jsonl'):
    p=json.loads(line); n+=1; t+=sum(x['num_tokens'] for x in p)
print(f'{n} pack, trung bình {t/n:.0f} token/pack')"
```

Mỗi trang A4 200 DPI (1656×2339 ≈ 3,87 Mpx, dưới ngưỡng `max_pixels`
= 2048² nên **không bị resize**) tốn phần lớn ngân sách token là vision token.
Nếu số pack ra quá ít (mỗi pack chỉ 2–3 trang) thì cân nhắc `--max-pixels`
nhỏ hơn khi pack — đổi lại chữ nhỏ trên giấy sẽ mờ đi.

## 3. Train

```bash
MODEL_PATH=$MODEL_PATH \
TRAIN_DATA=./data/packed/train_16384.jsonl \
PACK_LEN=16384 \
    bash scripts/sft_base_1gpu.sh
```

Khác gì so với `scripts/sft_base.sh`:

| | `sft_base.sh` | `sft_base_1gpu.sh` |
|---|---|---|
| GPU | 8 | **1** |
| Env | `env_common.sh` (IB, `bond1`, 8 HCA mlx5) | **`env_single.sh`** (không IB) |
| `packed_max_length` | 20480 | **16384** |
| `GRAD_ACCUM` | 1 | **4** |
| `SAVE_STEPS` | 200 | **50** |
| `warmup_ratio` | 0.03 | 0.03 |
| DeepSpeed | không | tuỳ chọn qua `DEEPSPEED=scripts/zero2.json` |

Model chỉ ~1B tham số nên ở bf16 + `gradient_checkpointing` thì 80GB thừa sức
chạy full SFT (vision + MLP + LLM) mà **không cần ZeRO**. Biến còn lại ăn VRAM là
`PACK_LEN`; upstream ghi rằng 20480 vẫn OOM được trên 80GB nên fork lấy 16384 làm
mặc định. Nếu `nvidia-smi` cho thấy còn nhiều headroom, nâng dần:

```bash
PACK_LEN=20480 bash scripts/sft_base_1gpu.sh    # nhớ pack lại data cùng độ dài
```

`PACK_LEN` lúc train **phải bằng** `PACK_LEN` lúc pack.

Các nút khác:

```bash
TUNE_VISION=False ...     # đóng băng vision tower, nhanh hơn ~15%
EPOCHS=3 LR=1e-5 ...
DEEPSPEED=scripts/zero2.json ...   # chỉ khi OOM sau khi đã hạ PACK_LEN
```

Train tự resume: nếu `output_dir` đã có `checkpoint-*` thì
`train_hunyuan.py:221` gọi `trainer.train(resume_from_checkpoint=True)`. Muốn
chạy lại từ đầu thì đổi `RUN_NAME` hoặc xoá thư mục.

## 4. Đánh giá

**Không bật `--eval_strategy steps`.** `train_hunyuan.py:198-205` tạo
`eval_dataset` với `is_packed=False` (trả về dict lẻ) nhưng collator lúc đó là
`PackedVLDataCollator` (chờ list-of-list) → vỡ ngay batch eval đầu tiên. Đánh giá
tách rời bằng `data/raw/validation.jsonl` và code inference trong `inference/`.

## 5. Những chỗ đã sửa của upstream

| File | Sửa |
|---|---|
| `scripts/sft_base.sh:106`, `sft_dflash.sh:127`, `sft_dflash_finetune.sh:140` | `${entry_file} "${args}"` → bỏ nháy. Có nháy thì toàn bộ flag thành **một** phần tử `argv`, `HfArgumentParser.parse_args_into_dataclasses()` chết ngay. |
| `tools/pipeline_count_and_pack.py` | thêm `--allow-empty-answer` |
| `scripts/pack_data.sh` | thêm `ALLOW_EMPTY`, `FOREGROUND` |

## 6. Ghi chú về quy mô

Bộ này có 1.007 trang train. Recipe của Tencent dùng ~1M pack cho pretrain và
14,7k pack cho domain finetune, tức là ta ít hơn khoảng hai bậc. Kỳ vọng thực tế
là **thích nghi domain** (đúng bố cục biểu mẫu VN, đúng quy ước Markdown trong
`ocr_prompt.md`), không phải nâng năng lực OCR tổng quát. Nếu loss train tụt
nhanh mà chất lượng trên validation không lên, đó là overfit — hạ `EPOCHS`
xuống 2–3 và/hoặc đóng băng vision tower trước khi nghĩ tới việc đổi LR.
