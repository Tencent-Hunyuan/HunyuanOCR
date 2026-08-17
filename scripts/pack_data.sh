#!/bin/bash
# ============================================================================
# Data packing pipeline for HunyuanOCR SFT / DFlash training.
#
# Reads a list of raw OCR JSONL files (configs/data_list.txt), tokenizes with
# the base model's tokenizer, then packs multiple samples up to a fixed max
# length into single sequences to maximize GPU utilization.
#
# Output: ./data/parsing_packed_${PACK_LEN}.jsonl
# ============================================================================

set -e

# ────────────── Config ──────────────
INPUT_LIST=${INPUT_LIST:-./configs/data_list.txt}
MODEL_PATH=${MODEL_PATH:-/path/to/HunyuanOCR/base/model}
COUNT_OUTPUT_DIR=${COUNT_OUTPUT_DIR:-./data/parsing_jsonl_count}
PACK_LEN=${PACK_LEN:-20480}
PACK_OUTPUT=${PACK_OUTPUT:-./data/parsing_packed_${PACK_LEN}.jsonl}
NUM_PROCESSES=${NUM_PROCESSES:-32}
THREADS_PER_PROCESS=${THREADS_PER_PROCESS:-8}
LOG_FILE=${LOG_FILE:-pack_data.log}
# Keep blank-page samples (empty answer). Set ALLOW_EMPTY=0 to drop them.
ALLOW_EMPTY=${ALLOW_EMPTY:-1}
EMPTY_FLAG=""
[ "$ALLOW_EMPTY" = "1" ] && EMPTY_FLAG="--allow-empty-answer"
# Run in the foreground when FOREGROUND=1 (easier to watch on one machine).
FOREGROUND=${FOREGROUND:-0}

# ────────────── Sanity check ──────────────
if [ ! -f "$INPUT_LIST" ]; then
    echo "[ERROR] input list not found: $INPUT_LIST"
    echo "Please fill in configs/data_list.txt with one raw JSONL path per line."
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] base model dir not found: $MODEL_PATH"
    echo "Please set MODEL_PATH to your HunyuanOCR base model directory."
    exit 1
fi

mkdir -p "$COUNT_OUTPUT_DIR"
mkdir -p "$(dirname "$PACK_OUTPUT")"

echo "========================================"
echo "  Input list       : $INPUT_LIST  ($(wc -l < "$INPUT_LIST") files)"
echo "  Model path       : $MODEL_PATH"
echo "  Count output dir : $COUNT_OUTPUT_DIR"
echo "  Pack output      : $PACK_OUTPUT"
echo "  Pack length      : $PACK_LEN"
echo "  Processes        : $NUM_PROCESSES × $THREADS_PER_PROCESS threads"
echo "  Log              : $LOG_FILE"
echo "========================================"

# ────────────── Run ──────────────
cmd=(python tools/pipeline_count_and_pack.py
    --input-list "$INPUT_LIST"
    --model-path "$MODEL_PATH"
    --count-output-dir "$COUNT_OUTPUT_DIR"
    --pack-output "$PACK_OUTPUT"
    --num-processes "$NUM_PROCESSES"
    --threads-per-process "$THREADS_PER_PROCESS"
    --pack-length "$PACK_LEN")
[ -n "$EMPTY_FLAG" ] && cmd+=("$EMPTY_FLAG")

if [ "$FOREGROUND" = "1" ]; then
    "${cmd[@]}" 2>&1 | tee "$LOG_FILE"
else
    nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "[started] pid=$PID  log=$LOG_FILE"
    echo "Monitor with:  tail -f $LOG_FILE"
fi
