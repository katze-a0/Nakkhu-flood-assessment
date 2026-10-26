#!/usr/bin/env bash
# Plan 014 on the VPS: every lane inside ONE resource scope (start this script under
#   systemd-run --scope -p CPUQuota=700% -p MemoryMax=11G -p MemorySwapMax=0 nice -n 19 ionice -c3)
# so that at most 7 single-threaded processes run at once.
#   bash vps_lanes.sh BASE
# Lanes: P1-4 | P1-10 then P1-9 | P1-3 then P1-7 | P1-5 then P1-8, next to P1-6 units 0-51/78
# (3 jobs); when the four quick lanes are done, P1-6 units 52-77/78 (4 jobs).
set -u
B=$(cd "$1" && pwd)
cd "$B"
lane() { for t in "$@"; do bash run.sh "$B" vps "$t"; done; }
echo "start $(date -u +%FT%TZ)" > logs/lanes.log
lane P1-4 & a=$!
lane P1-10 P1-9 & b=$!
lane P1-3 P1-7 & c=$!
lane P1-5 P1-8 & d=$!
bash run.sh "$B" vps P1-6 0-51/78 3 & e=$!
wait $a $b $c $d
echo "quick lanes done $(date -u +%FT%TZ)" >> logs/lanes.log
bash run.sh "$B" vps P1-6 52-77/78 4 & f=$!
wait $e $f
echo "all done $(date -u +%FT%TZ)" >> logs/lanes.log
