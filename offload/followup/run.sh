#!/usr/bin/env bash
# Plan 014 remote execution of one reserved test (or P1-6 shards) from its receipt.
#   bash run.sh BASE HOST_TYPE TEST [SHARD JOBS]
# Single-threaded BLAS/OpenMP per process. Log + exit code in BASE/logs/. The receipt is found
# in BASE/root/runs/followup/*/receipt.json by test id (exactly one).
set -uo pipefail
BASE=$(cd "$1" && pwd); HOST=$2; TEST=$3; SHARD=${4:-}; JOBS=${5:-1}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NAKKHU_HOST_TYPE="$HOST" PYTHONUNBUFFERED=1
cd "$BASE/root"
PYV="$BASE/root/.venv/bin/python"
REC=$("$PYV" - "$TEST" <<'EOF'
import json, sys, glob
hits = [p for p in sorted(glob.glob("runs/followup/*/receipt.json"))
        if json.load(open(p)).get("test") == sys.argv[1]]
print(hits[-1] if len(hits) == 1 else "")
EOF
)
TAG="$TEST${SHARD:+_$(echo "$SHARD" | tr '/,' '__')}"
LOG="$BASE/logs/$TAG.log"
if [ -z "$REC" ]; then echo "no unique receipt for $TEST" > "$LOG"; echo 2 > "$LOG.rc"; exit 2; fi
ARGS=(followup run --test "$TEST" --receipt "$REC")
if [ -n "$SHARD" ]; then ARGS+=(--shard "$SHARD" --jobs "$JOBS"); fi
echo "start $(date -u +%FT%TZ) $TEST $SHARD jobs=$JOBS" > "$LOG"
"$BASE/root/.venv/bin/nakkhu" "${ARGS[@]}" >> "$LOG" 2>&1
RC=$?
echo "end $(date -u +%FT%TZ) rc=$RC" >> "$LOG"
echo "$RC" > "$LOG.rc"
exit "$RC"
