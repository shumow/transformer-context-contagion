#!/usr/bin/env bash
# E2 induction-head attribution at scale, for a >=24-48GB GPU box (RunPod/Lambda/etc).
# Run from inside the cloned repo after bootstrap.sh. Wrap it for long runs:
#   tmux new -s e2                       # then: deploy/run_e2_scale.sh
#   # or detached:  nohup deploy/run_e2_scale.sh > /workspace/e2.log 2>&1 &
#
# Uses --k-frac (a FRACTION of heads) so the ablation set scales with model size -- a
# fixed 8 heads washed out the signal on gpt2-xl (8/1200 heads). See results_log.md (E2 at
# scale). E2 checkpoints per (p,N) cell and resumes, so this is safe on preemptible/Spot.
#
# Override any of these via env, e.g.:
#   MODELS="EleutherAI/pythia-6.9b" K_FRAC=0.08 deploy/run_e2_scale.sh
set -euo pipefail

cd "$(dirname "$0")/.."                          # repo root
[ -d .venv ] || { echo "no .venv here -- run bootstrap.sh first"; exit 1; }
. .venv/bin/activate
git pull --ff-only 2>/dev/null || echo "(git pull skipped -- using local code)"

# RunPod (and some images) set HF_HUB_ENABLE_HF_TRANSFER=1 but don't ship hf_transfer, which
# makes every HF download error out. Disable it if the package is missing -- the default
# download path is plenty fast.
python -c "import hf_transfer" 2>/dev/null || { export HF_HUB_ENABLE_HF_TRANSFER=0; echo "hf_transfer absent -> HF_HUB_ENABLE_HF_TRANSFER=0"; }

K_FRAC="${K_FRAC:-0.05}"                          # fraction of all heads to ablate
LENGTHS="${LENGTHS:-1 3 5}"
REPS="${REPS:-2 3 4 8 16}"
PAYLOADS="${PAYLOADS:-4}"
GET_PYTHIA69="${GET_PYTHIA69:-1}"                 # pull pythia-6.9b (clean 7B, same family as the 1.4B confirmation)
# Default ladder: gpt2-xl re-run + Pythia 2.8B/6.9B (both OOM'd or absent on the T4) + a
# Qwen2.5-7B attempt (TransformerLens may not support it -- it'll just skip if so).
MODELS="${MODELS:-gpt2-xl EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b Qwen/Qwen2.5-7B}"

if [ "$GET_PYTHIA69" = "1" ]; then
  echo "==> fetching EleutherAI/pythia-6.9b (clean 7B for the Pythia E2 ladder)"
  hf download EleutherAI/pythia-6.9b >/dev/null 2>&1 || echo "   (download failed -- will skip if absent)"
fi

for m in $MODELS; do
  echo "===================== E2 $m  (k-frac=$K_FRAC) ====================="
  python -m experiments.e2_induction --model "$m" --device cuda \
    --lengths $LENGTHS --reps $REPS --payloads "$PAYLOADS" --k-frac "$K_FRAC" \
    || echo "!! E2 failed for $m (TransformerLens support? OOM?) -- continuing"
done
echo "===== E2 SCALE DONE ====="
echo "results in results/e2_*.json  (fetch them before terminating the pod)"
