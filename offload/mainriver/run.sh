#!/usr/bin/env bash
# Main-river recompute, one remote step, single-threaded BLAS/OpenMP; log + rc in BASE/logs/.
#   bash run.sh BASE HOST_TYPE surfaces
#   bash run.sh BASE HOST_TYPE check HAND_RUN
#   bash run.sh BASE HOST_TYPE exposure|event|roc    (receipt found by analysis, exactly one)
set -uo pipefail
BASE=$(cd "$1" && pwd); HOST=$2; STEP=$3; ARG=${4:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NAKKHU_HOST_TYPE="$HOST" PYTHONUNBUFFERED=1
cd "$BASE/root"
NK="$BASE/root/.venv/bin/nakkhu"
LOG="$BASE/logs/$STEP.log"
echo "start $(date -u +%FT%TZ) $STEP $ARG" > "$LOG"
case "$STEP" in
  surfaces) "$NK" mainriver surfaces >> "$LOG" 2>&1 ;;
  check) "$NK" mainriver check-inputs --hand-run "$ARG" > "$BASE/logs/check.json" 2>> "$LOG" ;;
  exposure|event|roc)
    REC=$("$BASE/root/.venv/bin/python" - "$STEP" <<'PYEOF'
import json, sys, glob
hits = [p for p in sorted(glob.glob("runs/mainriver/*/receipt.json"))
        if json.load(open(p)).get("analysis") == sys.argv[1]]
print(hits[-1] if len(hits) == 1 else "")
PYEOF
)
    if [ -z "$REC" ]; then echo "no unique receipt for $STEP" >> "$LOG"; echo 2 > "$LOG.rc"; exit 2; fi
    "$NK" mainriver run --analysis "$STEP" --receipt "$REC" >> "$LOG" 2>&1 ;;
  *) echo "unknown step $STEP" >> "$LOG"; echo 2 > "$LOG.rc"; exit 2 ;;
esac
RC=$?
echo "end $(date -u +%FT%TZ) rc=$RC" >> "$LOG"
echo "$RC" > "$LOG.rc"
exit "$RC"
