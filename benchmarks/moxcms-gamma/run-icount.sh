#!/bin/bash
# Instruction counts for run_work only. Usage: run-icount.sh <icount-binary> <variants...>
set -euo pipefail
bin=${1:?supply the icount binary}; shift
for load in "eval_encode 0" "eval_encode 10" "eval_encode 50" "eval_encode 100" "eval_decode 0" "eval_decode 50" "p3_adobe_f32 0" "srgb_adobe_f32 0" "p3_adobe_u8 0"; do
  for v in "$@"; do
    out=$(valgrind --tool=callgrind --callgrind-out-file=/dev/null '--toggle-collect=*run_work*' --branch-sim=yes "$bin" $v $load 4096 2>&1)
    ir=$(grep -E "Collected :" <<<"$out" | sed 's/.*Collected : //')
    echo "$load $v collected: $ir"
  done
done
