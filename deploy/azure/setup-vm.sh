#!/usr/bin/env bash
# Provision the Track B GPU VM: clone the repo, build a CUDA venv, install the
# mechanistic-tier deps, and pre-download the models to the (persistent) OS disk so
# subsequent sessions start instantly. Run this ONCE, on the VM, after `up.sh`:
#
#   scp -i ~/.ssh/tcc_a10 deploy/azure/setup-vm.sh azureuser@<IP>:~
#   ssh  -i ~/.ssh/tcc_a10 azureuser@<IP> 'HF_TOKEN=hf_xxx bash ~/setup-vm.sh'
#
# HF_TOKEN is optional -- without it the ungated models still download; the gated
# ones (Llama-3.2, Gemma-2) are skipped. Set it to also fetch those (accept each
# model's license on huggingface.co first).
set -euo pipefail

REPO="${REPO:-https://github.com/shumow/transformer-context-contagion.git}"
DIR="${DIR:-$HOME/transformer-context-contagion}"
WITH_VLLM="${WITH_VLLM:-0}"   # set 1 to also install vLLM (heavy; only needed for big throughput sweeps)

echo "==> nvidia-smi"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "==> ensure python venv + pip (ubuntu-hpc base image lacks ensurepip)"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip >/dev/null

echo "==> clone $REPO"
[ -d "$DIR/.git" ] || git clone "$REPO" "$DIR"
cd "$DIR"

echo "==> python venv + CUDA torch"
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
# CUDA build of torch (cu121 wheels are broadly driver-compatible); then the rest.
pip install -q --index-url https://download.pytorch.org/whl/cu121 torch
# nnsight omitted -- unused in the repo (only transformer_lens) and slow to resolve.
# Constrain torch to the CUDA build just installed so pip doesn't re-download a second torch.
pip install -q numpy matplotlib "transformers>=4.44" "accelerate>=0.33" \
               "transformer_lens>=2.0" huggingface_hub \
               -c <(pip freeze | grep -iE '^torch==')
[ "$WITH_VLLM" = "1" ] && pip install -q vllm || true

echo "==> sanity: torch sees the GPU"
python -c "import torch;assert torch.cuda.is_available();print('CUDA OK:',torch.cuda.get_device_name(0),torch.__version__)"

# --- model pre-download (cached under ~/.cache/huggingface, on the persistent OS disk) ---
# Ungated: download always. Covers the E1.3 size axis 355M -> 7B plus instruct (E1.4/E8).
UNGATED=(
  gpt2-medium gpt2-large gpt2-xl
  EleutherAI/pythia-410m EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b
  Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B
  Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct
)
# Gated: need HF_TOKEN + accepted license on huggingface.co.
GATED=(
  meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B meta-llama/Llama-3.2-3B-Instruct
  google/gemma-2-2b google/gemma-2-2b-it
)

# huggingface_hub >=1.0 renamed the CLI to `hf` and dropped `--quiet` on download.
dl(){ echo "   -- $1"; hf download "$1" >/dev/null 2>&1 || echo "      FAILED: $1"; }

echo "==> pre-downloading ungated models"
for m in "${UNGATED[@]}"; do dl "$m"; done

if [ -n "${HF_TOKEN:-}" ]; then
  echo "==> HF_TOKEN set -- pre-downloading gated models"
  export HF_TOKEN
  for m in "${GATED[@]}"; do dl "$m"; done
else
  echo "==> HF_TOKEN not set -- skipping gated models (Llama-3.2, Gemma-2)"
fi

echo "==> done. activate with:  cd $DIR && . .venv/bin/activate"
echo "    e.g.  python -m experiments.e1_3_size_sweep --models gpt2-medium gpt2-large Qwen/Qwen2.5-7B --device cuda"
