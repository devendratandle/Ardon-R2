#!/usr/bin/env bash
# Differential correctness harness: run each case under both Ardon-R2 and
# GNU R, extract `key=value` lines, and compare numerically.
#
#   tests/differential/run.sh [case-name-filter]
#
# A case is a single .R file in cases/ that is valid in BOTH engines and
# prints only deterministic `key=value` lines via cat(). Values compare
# equal within a relative tolerance of 1e-9 (absolute 1e-12 near zero).
# String values must match exactly. A key printed by only one engine is a
# failure. RNG-dependent values must not be emitted (the rnorm/runif
# streams differ between engines by design) — derive structural facts
# (lengths, dims, classes) from random data instead.
#
# Exit code: 0 if every case passes, 1 otherwise.

set -u
cd "$(dirname "$0")"

R2_BIN="${R2_BIN:-../../target/release/r2.exe}"
RSCRIPT="${RSCRIPT:-C:/R/R-4.5.3/bin/x64/Rscript.exe}"
FILTER="${1:-}"

[ -x "$R2_BIN" ] || { echo "r2 binary not found: $R2_BIN (set R2_BIN)"; exit 1; }
command -v "$RSCRIPT" >/dev/null 2>&1 || [ -x "$RSCRIPT" ] || { echo "Rscript not found: $RSCRIPT (set RSCRIPT)"; exit 1; }

extract_kv() {  # stdin -> key=value lines only, CR stripped
    tr -d '\r' | grep -E '^[A-Za-z][A-Za-z0-9_.]*=' || true
}

pass=0; fail=0; failed_cases=""
for case_file in cases/*.R; do
    name=$(basename "$case_file" .R)
    [ -n "$FILTER" ] && case "$name" in *"$FILTER"*) ;; *) continue;; esac

    r2_out=$("$R2_BIN" "$case_file" 2>&1 | extract_kv)
    r_out=$("$RSCRIPT" "$case_file" 2>&1 | extract_kv)

    diff_report=$(awk -F= '
        NR==FNR { a[$1]=substr($0, length($1)+2); next }
        {
            k=$1; v=substr($0, length($1)+2)
            if (!(k in a)) { print "  R-only key: " k; bad=1; next }
            x=a[k]; seen[k]=1
            # numeric compare when both parse as numbers, else exact string
            if (x+0 == x && v+0 == v) {
                dx=x+0; dv=v+0; d=dx-dv; if (d<0) d=-d
                m=(dx<0?-dx:dx); u=(dv<0?-dv:dv); if (u>m) m=u
                tol=m*1e-9; if (tol<1e-12) tol=1e-12
                if (d>tol) { printf "  MISMATCH %s: r2=%s R=%s\n", k, x, v; bad=1 }
            } else if (x != v) { printf "  MISMATCH %s: r2=%s R=%s\n", k, x, v; bad=1 }
        }
        END { for (k in a) if (!(k in seen)) { print "  r2-only key: " k; bad=1 }
              exit bad }
    ' <(printf '%s\n' "$r2_out") <(printf '%s\n' "$r_out"))

    if [ $? -eq 0 ] && [ -n "$r2_out" ]; then
        pass=$((pass+1)); echo "PASS  $name"
    else
        fail=$((fail+1)); failed_cases="$failed_cases $name"
        echo "FAIL  $name"
        [ -z "$r2_out" ] && echo "  (r2 produced no key=value output)"
        printf '%s\n' "$diff_report"
    fi
done

echo
echo "differential: $pass passed, $fail failed"
[ -n "$failed_cases" ] && echo "failed:$failed_cases"
[ $fail -eq 0 ]
