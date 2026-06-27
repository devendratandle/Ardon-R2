#!/usr/bin/env bash
# Linux/macOS equivalent of run.ps1.
#
# Prereqs:
#   - `Rscript` on PATH (CRAN R 4.x+)
#   - r2 built in release mode (run `cargo build --release` from repo root)
#
# Usage from repo root:
#   bash bench/r_vs_r2/run.sh

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
r2_exe="$repo_root/target/release/r2"

if [[ ! -x "$r2_exe" ]]; then
    echo "ERROR: r2 not found at $r2_exe. Run 'cargo build --release' first." >&2
    exit 1
fi
if ! command -v Rscript >/dev/null 2>&1; then
    echo "ERROR: Rscript not on PATH. Install R or add it to PATH." >&2
    exit 1
fi

run_pair() {
    local name="$1" rScript="$2" r2Script="$3"
    printf "\n═══ %s ═══\n" "$name"
    echo "Running R:  $rScript"
    local r_out;  r_out="$(Rscript --vanilla "$rScript" 2>/dev/null)"
    echo "Running R2: $r2Script"
    local r2_out; r2_out="$("$r2_exe" "$r2Script" 2>/dev/null)"

    # Parse `KEY=VALUE` (R uses `cat` so KEY=VALUE is bare; R2 uses `print(paste0)`
    # which wraps in `[1] "..."` quotes — strip them).
    extract() {
        echo "$1" | sed -E 's/^\s*\[?[0-9]*\]?\s*"?([A-Za-z0-9_.]+)=([^"]*)"?\s*$/\1=\2/' | grep -E '^[A-Za-z0-9_.]+='
    }

    declare -A R_MAP R2_MAP
    while IFS='=' read -r k v; do R_MAP["$k"]="$v"; done < <(extract "$r_out")
    while IFS='=' read -r k v; do R2_MAP["$k"]="$v"; done < <(extract "$r2_out")

    printf "%-30s %-22s %-22s %-12s\n" "key" "R" "R2" "delta"
    printf "%-30s %-22s %-22s %-12s\n" "------------------------------" "----------------------" "----------------------" "------------"

    # Union of keys, sorted.
    local keys; keys=$(printf "%s\n%s\n" "${!R_MAP[@]}" "${!R2_MAP[@]}" | sort -u)
    for k in $keys; do
        local rv="${R_MAP[$k]:-<missing>}"
        local r2v="${R2_MAP[$k]:-<missing>}"
        local delta=""
        if [[ "$rv" =~ ^-?[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?$ ]] && \
           [[ "$r2v" =~ ^-?[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?$ ]]; then
            delta=$(awk -v a="$rv" -v b="$r2v" 'BEGIN{
                d = (a-b); if (d<0) d=-d;
                if (a != 0) { printf "%.3g (%.2f%%)", d, 100*d/(a<0?-a:a); }
                else        { printf "%.3g",         d; }
            }')
        fi
        printf "%-30s %-22s %-22s %-12s\n" "$k" "$rv" "$r2v" "$delta"
    done
}

run_pair "ACCURACY" "$script_dir/accuracy.R" "$script_dir/accuracy.r2"
run_pair "PERFORMANCE (seconds, lower is better)" "$script_dir/performance.R" "$script_dir/performance.r2"

printf "\nDone. Accuracy deltas under ~1e-3 are typically within R2's stated tolerance.\n"
printf "Performance: R's matrix-multiply depends on the BLAS R is linked to.\n"
