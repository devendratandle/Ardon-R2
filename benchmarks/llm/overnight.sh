#!/usr/bin/env bash
# One long complete run, R2 then PyTorch: the 7.24M model (dim 256, 4 layers,
# vocab 8000) trained for several epochs on corpus.txt, with held-out loss on
# both engines every 1,000 steps. About 2 hours on R2 and 3 on PyTorch on the
# AMD laptop. Results: ts_run/overnight/results.txt.
#
#   bash benchmarks/llm/overnight.sh
#   R2_STEPS=20 R2_EVAL_EVERY=10 bash benchmarks/llm/overnight.sh    # to test this script
#
# Build first (and run no cargo while this runs):
#   CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
#     cargo build --release -p r2-train --example tinystories_train

set -u
cd "$(dirname "$0")/../.."

X=./target/release/examples/tinystories_train.exe
PY="${PY:-C:/Users/deven/AppData/Local/Programs/Python/Python312/python.exe}"
OUT=ts_run/overnight
[ -x "$X" ] || { echo "not built: $X"; exit 1; }
[ -x "$PY" ] || { echo "no python at $PY (set PY)"; exit 1; }
mkdir -p "$OUT"
R="$OUT/results.txt"

export R2_DIM=256 R2_LAYERS=4 R2_FFN=768 R2_HEADS=4 R2_KV=2 R2_VOCAB=8000
export R2_SEQ=256 R2_BATCH=8 R2_LR=1e-3 R2_SCHED=cosine R2_WARMUP=200 R2_CLIP=1
export R2_STEPS="${R2_STEPS:-18000}" R2_EVAL_EVERY="${R2_EVAL_EVERY:-1000}" R2_GEN=64

clock() { powershell -NoProfile -Command '(Get-CimInstance Win32_Processor).CurrentClockSpeed' | tr -d '\r'; }

{ echo; echo "=== $(date '+%Y-%m-%d %H:%M')  $(git log --oneline -1 | cut -c1-60)  $R2_STEPS steps"; } >> "$R"
echo "R2 starts $(date +%H:%M) at $(clock) MHz" | tee -a "$R"
"$X" > "$OUT/r2.log" 2>&1 || { echo "R2 FAILED, see $OUT/r2.log" | tee -a "$R"; exit 1; }
rm -rf "$OUT/r2_model"
cp -r ts_run/model "$OUT/r2_model" && cp ts_run/tokenizer.json "$OUT/r2_model/"

echo "PyTorch starts $(date +%H:%M) at $(clock) MHz" | tee -a "$R"
"$PY" benchmarks/llm/tinystories_train.py > "$OUT/pytorch.log" 2>&1 \
    || { echo "PyTorch FAILED, see $OUT/pytorch.log" | tee -a "$R"; exit 1; }
echo "finished $(date +%H:%M) at $(clock) MHz" | tee -a "$R"
sed -n '/^  tokens /,$p' "$OUT/pytorch.log" | tee -a "$R"
