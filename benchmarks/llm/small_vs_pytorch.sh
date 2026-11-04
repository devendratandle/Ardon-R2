#!/usr/bin/env bash
# A small model trained to the end of its schedule, on R2 and on PyTorch: the
# whole pipeline, with speed, tuning, context and accuracy compared at each
# stage. Every leg runs on both, from the same weights on the same tokens.
#
#   bash benchmarks/llm/small_vs_pytorch.sh            # about 45 minutes
#   SMOKE=1 bash benchmarks/llm/small_vs_pytorch.sh    # every leg at 20 steps, to test this script
#
# The model is 1.30M parameters (dim 128, 4 layers, vocab 2000) on the first
# 5 MB of corpus.txt, 1,024 tokens per step throughout. The final run is EPOCHS
# passes over the training tokens (default 8), cut short only if it would pass
# MINUTES (default 58). Results: ts_run/small/results.txt.
#
# Build first (and run no cargo while this runs):
#   CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
#     cargo build --release -p r2-train --example tinystories_train

set -u
cd "$(dirname "$0")/../.."

X=./target/release/examples/tinystories_train.exe
PY="${PY:-C:/Users/deven/AppData/Local/Programs/Python/Python312/python.exe}"
OUT=ts_run/small
EPOCHS="${EPOCHS:-8}"
MINUTES="${MINUTES:-58}"
[ -x "$X" ] || { echo "not built: $X"; exit 1; }
[ -x "$PY" ] || { echo "no python at $PY (set PY)"; exit 1; }
[ -f corpus.txt ] || { echo "no corpus.txt"; exit 1; }
mkdir -p "$OUT"
R="$OUT/results.txt"

scale() { if [ "${SMOKE:-0}" = 1 ]; then echo 20; else echo "$1"; fi; }

# whole lines only, so no sentence or character is cut
export R2_CORPUS="$OUT/corpus_5mb.txt"
head -c 5000000 corpus.txt | sed '$d' > "$R2_CORPUS"
export R2_VOCAB=2000 R2_DIM=128 R2_LAYERS=4 R2_FFN=384 R2_HEADS=4 R2_KV=2 R2_GEN=48 R2_VAL_BATCHES=26
# from raw text: the first leg learns the vocabulary and tokenizes, on the record
rm -f ts_run/tokens.stamp

row() { printf '%-22s %9s %9s %8s %8s %9s %9s %7s\n' "$@" | tee -a "$R"; }
header() {
    echo | tee -a "$R"
    echo "--- $1" | tee -a "$R"
    row leg "R2 loss" "PT loss" "R2 b/B" "PT b/B" "R2 ms" "PT ms" faster
}

# pair <name> <steps> [VAR=value ...]: the run on R2, then on PyTorch, which
# reads R2's manifest and so trains the same recipe from the same start.
pair() {
    local name=$1 steps=$2; shift 2
    env R2_STEPS="$steps" "$@" "$X" > "$OUT/$name.r2.log" 2>&1 \
        || { echo "$name: R2 FAILED, see $OUT/$name.r2.log" | tee -a "$R"; return 1; }
    "$PY" benchmarks/llm/tinystories_train.py > "$OUT/$name.pt.log" 2>&1 \
        || { echo "$name: PyTorch FAILED, see $OUT/$name.pt.log" | tee -a "$R"; return 1; }
    read -r lr lp br bp mr mp < <(awk '
        $1 == "ms/step" { mr = $2; mp = $3 }
        /^loss after training/ { lr = $4; lp = $5 }
        /^bits per byte/ { br = $4; bp = $5 }
        END { print lr, lp, br, bp, mr, mp }' "$OUT/$name.pt.log")
    row "$name" "$lr" "$lp" "$br" "$bp" "$mr" "$mp" "$(awk -v a="$mr" -v b="$mp" 'BEGIN { printf "%.2fx", b / a }')"
    echo "$name $lr $lp $mr $mp" >> "$OUT/table.txt"
}

# the leg among those named with the lowest loss, in column 2 (R2) or 3 (PyTorch)
best() { local col=$1; shift; for n in "$@"; do grep "^$n " "$OUT/table.txt"; done | sort -k"$col" -g | head -1 | cut -d' ' -f1; }

: > "$OUT/table.txt"
{ echo; echo "=== $(date '+%Y-%m-%d %H:%M')  $(git log --oneline -1 | cut -c1-60)"; } >> "$R"
TUNE="R2_SEQ=64 R2_BATCH=16"
RECIPE="R2_SCHED=cosine R2_WARMUP=30 R2_CLIP=1"

header "tuning and speed: 300 steps each, context 64"
pair lr_3e-4         "$(scale 300)" $TUNE R2_LR=3e-4
{
    echo
    echo "from raw text (first leg):"
    grep -E "tokenizer  |tokenize  " "$OUT/lr_3e-4.r2.log" | sed 's/^ */  R2  /' | cut -c1-90
    grep -E "^tokenize corpus|^PIPELINE TOTAL" "$OUT/lr_3e-4.pt.log" | sed 's/^/  /' | cut -c1-100
    echo
} | tee -a "$R"
pair lr_1e-3         "$(scale 300)" $TUNE R2_LR=1e-3
pair lr_3e-3         "$(scale 300)" $TUNE R2_LR=3e-3
pair lr_3e-3_cos_clip "$(scale 300)" $TUNE R2_LR=3e-3 $RECIPE
pair lr_3e-3_cos_clip_wd "$(scale 300)" $TUNE R2_LR=3e-3 $RECIPE R2_WD=0.01
TUNED=$(best 2 lr_3e-4 lr_1e-3 lr_3e-3 lr_3e-3_cos_clip lr_3e-3_cos_clip_wd)
case "$TUNED" in *3e-4*) LR=3e-4 ;; *1e-3*) LR=1e-3 ;; *) LR=3e-3 ;; esac
echo "best: R2 $TUNED, PyTorch $(best 3 lr_3e-4 lr_1e-3 lr_3e-3 lr_3e-3_cos_clip lr_3e-3_cos_clip_wd)" | tee -a "$R"

header "context: 500 steps each, same tokens per step, lr $LR, cosine, clip 1"
pair ctx_32  "$(scale 500)" R2_SEQ=32  R2_BATCH=32 R2_LR="$LR" $RECIPE
pair ctx_64  "$(scale 500)" R2_SEQ=64  R2_BATCH=16 R2_LR="$LR" $RECIPE
pair ctx_128 "$(scale 500)" R2_SEQ=128 R2_BATCH=8  R2_LR="$LR" $RECIPE
pair ctx_256 "$(scale 500)" R2_SEQ=256 R2_BATCH=4  R2_LR="$LR" $RECIPE
CTX=$(best 2 ctx_32 ctx_64 ctx_128 ctx_256)
SEQ=${CTX#ctx_}
BN=$((1024 / SEQ))
echo "best: R2 $CTX, PyTorch $(best 3 ctx_32 ctx_64 ctx_128 ctx_256)" | tee -a "$R"

# EPOCHS passes, unless what is left of MINUTES at these speeds buys fewer.
TOKENS=$(( $(wc -c < ts_run/train_ids.bin) / 4 ))
WANT=$((EPOCHS * TOKENS / 1024))
read -r MS < <(grep "^$CTX " "$OUT/table.txt" | awk '{ print $4 + $5 }')
LEFT=$((MINUTES * 60 - SECONDS - 60))
CAN=$(awk -v s="$LEFT" -v m="$MS" 'BEGIN { printf "%d", s * 1000 / m }')
STEPS=$(( WANT < CAN ? WANT : CAN ))
[ "$STEPS" -lt 100 ] && STEPS=100
echo "$((SECONDS / 60)) min used; the final run is $STEPS steps, $(awk -v s="$STEPS" -v t="$TOKENS" 'BEGIN { printf "%.1f", s * 1024 / t }') epochs of $TOKENS tokens" | tee -a "$R"

header "accuracy: the complete run, context $SEQ, lr $LR, cosine to lr/10, clip 1"
pair final "$(scale "$STEPS")" R2_SEQ="$SEQ" R2_BATCH="$BN" R2_LR="$LR" R2_SCHED=cosine R2_WARMUP=100 R2_CLIP=1
{
    echo
    grep -E "^train loss, first|^perplexity" "$OUT/final.pt.log"
    sed -n '/loss curve/,/^$/p' "$OUT/final.pt.log"
    grep -E "^  (prompt|R2|PyTorch) " "$OUT/final.pt.log"
    grep -E "^  (TRAINING|ACCURACY) " "$OUT/final.pt.log"
} | tee -a "$R"
rm -rf "$OUT/final_model"
cp -r ts_run/model "$OUT/final_model" && cp ts_run/tokenizer.json "$OUT/final_model/"
echo | tee -a "$R"
echo "total $((SECONDS / 60)) min; R2's model is in $OUT/final_model (llm.load reads it)" | tee -a "$R"
