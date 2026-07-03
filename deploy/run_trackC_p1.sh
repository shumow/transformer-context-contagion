#!/usr/bin/env bash
# Track C / P1 -- firm up the two thin-n E6 reversals with higher n + seed/seeding sweeps.
# Both reversals come from E6 (sampled, temperature 1.0), so more trials + more seeds is the
# whole game. Runs unattended -- launch it detached and walk away:
#
#   cd /workspace/tcc
#   nohup deploy/run_trackC_p1.sh > /workspace/trackC_p1.log 2>&1 &
#   disown                        # survives logout
#   tail -f /workspace/trackC_p1.log   # (optional) watch it
#
# Part A (worm-worse-at-scale): 7B serial passage, seeds x higher trials -- the SLOW half.
# Part B (partial firebreak):   5x5 cross-tokenizer matrix, swept over seeding STRENGTH
#   (--host0-reps) so the one-point "partial" claim becomes a curve: where does the firebreak
#   go from clean (weak seed, Track-A) to partial (strong seed)? Small models, cheap.
#
# Writes NEW prefixes (results/e6A_*, results/e6B_*) so the committed data/trackC snapshot is
# untouched. Each run also tees its own console to results/<prefix>.log. One run failing (OOM,
# gated weight, download stall) is logged and skipped -- the rest continue.
#
# Override any knob via env, e.g.:  SEEDS="7 8" TRIALS_A=10 REPS_SWEEP="16 32" deploy/run_trackC_p1.sh
set -uo pipefail                                   # NB: no -e; we want to survive a failed run

cd "$(dirname "$0")/.."                             # repo root
[ -d .venv ] || { echo "no .venv here -- run deploy/bootstrap.sh first"; exit 1; }
. .venv/bin/activate
git pull --ff-only 2>/dev/null || echo "(git pull skipped -- using local code)"

# RunPod images often set HF_HUB_ENABLE_HF_TRANSFER=1 without shipping hf_transfer, which makes
# every download error out. Disable it if the package is missing (default path is plenty fast).
python -c "import hf_transfer" 2>/dev/null || { export HF_HUB_ENABLE_HF_TRANSFER=0; echo "hf_transfer absent -> HF_HUB_ENABLE_HF_TRANSFER=0"; }

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-7 8 9}"                              # Part A: seeds (sampled runs need several)
TRIALS_A="${TRIALS_A:-20}"                           # Part A: trials/cell (was 5 in the snapshot)
REPS_SWEEP="${REPS_SWEEP:-8 16 32}"                  # Part B: seeding strength (host0-reps) to sweep
TRIALS_B="${TRIALS_B:-30}"                           # Part B: trials/cell (small models -> cheap)
mkdir -p results

ts() { date +'%H:%M:%S'; }
run() {                                             # run <prefix> <args...>
  local prefix="$1"; shift
  echo "[$(ts)] >>> $prefix  ::  $*"
  python -m experiments.e6_scale "$@" --device "$DEVICE" --prefix "results/$prefix" \
    > "results/$prefix.log" 2>&1 \
    && echo "[$(ts)] <<< $prefix OK  -> results/$prefix.json" \
    || echo "[$(ts)] !!! $prefix FAILED (see results/$prefix.log) -- continuing"
}

echo "=================== Track C / P1  ($(date)) ==================="
echo "device=$DEVICE  seeds=[$SEEDS] trials_A=$TRIALS_A  reps_sweep=[$REPS_SWEEP] trials_B=$TRIALS_B"

echo "--- Part A: worm at scale (7B serial, higher n) ---"
for s in $SEEDS; do
  run "e6A_hiN_s$s" --part A --seed "$s" --trials "$TRIALS_A"
done

echo "--- Part B: cross-tokenizer firebreak vs seeding strength ---"
for reps in $REPS_SWEEP; do
  run "e6B_reps$reps" --part B --host0-reps "$reps" --trials "$TRIALS_B"
done

echo "=================== Track C / P1 DONE  ($(date)) ==================="
echo "results: results/e6A_hiN_s*.json  results/e6B_reps*.json  (+ matching .log/.png)"
echo
echo "Snapshot them from your Mac before tearing the pod down, e.g.:"
echo "  POD=root@<ip> PORT=<port> KEY=~/.ssh/<key> RREPO=$(pwd) \\"
echo "    PATTERNS=\"$(for s in $SEEDS; do printf 'e6A_hiN_s%s ' "$s"; done)$(for r in $REPS_SWEEP; do printf 'e6B_reps%s ' "$r"; done)\" \\"
echo "    deploy/fetch_results.sh"
