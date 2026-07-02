#!/usr/bin/env bash
# Provider-agnostic bootstrap for the transformer-context-contagion Track B harness.
# Works on ANY Ubuntu + NVIDIA-CUDA GPU box (RunPod, Lambda, Vast.ai, Paperspace, a bare
# VM, ...). Unlike deploy/azure/setup-vm.sh it has NO cloud-specific bits -- no managed
# identity, no idle auto-deallocate -- because specialist GPU clouds bill per-second and
# you just terminate the box when done.
#
# Usage (on the GPU box):
#   bash bootstrap.sh                          # ungated models only
#   HF_TOKEN=hf_xxx bash bootstrap.sh          # also pull gated (Llama-3.2, Gemma-2)
#   DIR=/workspace/tcc bash bootstrap.sh       # clone to a persistent volume (e.g. RunPod /workspace)
#   TORCH_INDEX=.../cu124 bash bootstrap.sh    # if the box's driver needs a different CUDA wheel
#
# Or one-liner from a fresh box:
#   curl -sL https://raw.githubusercontent.com/shumow/transformer-context-contagion/main/deploy/bootstrap.sh | bash
set -euo pipefail

REPO="${REPO:-https://github.com/shumow/transformer-context-contagion.git}"
DIR="${DIR:-$HOME/transformer-context-contagion}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"  # broadly driver-compatible
WITH_VLLM="${WITH_VLLM:-0}"

# root in most GPU containers (RunPod/Vast); sudo on VM images (Lambda/Paperspace).
SUDO=""; [ "$(id -u)" != 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"

echo "==> GPU / driver"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || {
  echo "!! nvidia-smi not found -- this box has no usable NVIDIA GPU. Aborting."; exit 1; }

echo "==> base packages (git, python venv/pip)"
if command -v apt-get >/dev/null 2>&1 && { [ "$(id -u)" = 0 ] || [ -n "$SUDO" ]; }; then
  $SUDO apt-get update -qq || true
  # `env` (not a bare VAR= prefix) so an empty $SUDO doesn't make bash treat the
  # assignment as the command name.
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git python3-venv python3-pip >/dev/null || true
fi

echo "==> clone $REPO -> $DIR"
[ -d "$DIR/.git" ] || git clone "$REPO" "$DIR"
cd "$DIR"

echo "==> venv + CUDA torch (index: $TORCH_INDEX)"
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q --index-url "$TORCH_INDEX" torch
# nnsight is intentionally omitted -- nothing in the repo imports it (only transformer_lens),
# and it drags a heavy dep stack that makes pip backtrack for many minutes.
# Constrain torch to the CUDA build just installed, so this resolve does NOT re-download a
# second (default-index) torch -- that's a 500+ MB pull and can clobber the CUDA build.
pip install -q numpy matplotlib "transformers>=4.44" "accelerate>=0.33" \
               "transformer_lens>=2.0" "huggingface_hub>=0.34" \
               -c <(pip freeze | grep -iE '^torch==')
[ "$WITH_VLLM" = "1" ] && pip install -q vllm || true

echo "==> sanity: can torch see the GPU?"
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("CUDA:", ok, "|", (torch.cuda.get_device_name(0) if ok else "no device"), "| torch", torch.__version__)
if not ok:
    print("!! torch cannot see CUDA. Re-run with TORCH_INDEX matching the box's driver, e.g.")
    print("   TORCH_INDEX=https://download.pytorch.org/whl/cu118  (older)  or  .../cu124  (newer).")
    raise SystemExit(1)
PY

# --- model pre-download (HF cache; set HF_HOME to a persistent volume to keep it across boxes) ---
UNGATED=(
  gpt2-medium gpt2-large gpt2-xl
  EleutherAI/pythia-410m EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b
  Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B
  Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct
)
GATED=(
  meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B meta-llama/Llama-3.2-3B-Instruct
  google/gemma-2-2b google/gemma-2-2b-it
)
# hf (huggingface_hub >=1.0) with a fallback to the older huggingface-cli name.
dl(){ echo "   -- $1"; { hf download "$1" || huggingface-cli download "$1"; } >/dev/null 2>&1 \
        || echo "      FAILED: $1"; }

echo "==> pre-downloading ungated models"
for m in "${UNGATED[@]}"; do dl "$m"; done
if [ -n "${HF_TOKEN:-}" ]; then
  echo "==> HF_TOKEN set -- pulling gated models"; export HF_TOKEN
  for m in "${GATED[@]}"; do dl "$m"; done
else
  echo "==> no HF_TOKEN -- skipping gated (Llama-3.2, Gemma-2). Re-run with HF_TOKEN=hf_xxx to add them."
fi

cat <<EOF

Ready.  cd $DIR && . .venv/bin/activate
  E1.3 size sweep (355M-7B):
    python -m experiments.e1_3_size_sweep --models gpt2-xl Qwen/Qwen2.5-7B \\
        --lengths 1 3 5 8 --reps 1 2 4 8 16 32 --device cuda --dtype float16
  E2 induction attribution (a >=24GB GPU handles 7B; checkpoints per cell + resumes):
    python -m experiments.e2_induction --model Qwen/Qwen2.5-7B --device cuda \\
        --lengths 1 3 5 --reps 2 3 4 8 16 --payloads 4 --k-heads 8
  For long runs, detach:  nohup python -m experiments.e2_induction ... > run.log 2>&1 &

  COST: specialist GPU clouds bill for as long as the box exists and there is NO
  idle auto-deallocate here -- TERMINATE the instance when you're done.
EOF
