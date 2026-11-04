#!/usr/bin/env bash
# Unattended pod job for Tier 2 (review_remediation_plan.md 2.1-2.3). Runs INSIDE a RunPod
# pod as its start command (deploy/runpod_pod.py creates the pod with this as dockerArgs via
# a git clone). Everything is driven by env:
#
#   TCC_JOB             diag | smoke | p1 | e4e5 | followup   (what to run; see run_job below)
#   TCC_BRANCH          code branch to run         (the clone is made by the start command)
#   TCC_RESULTS_BRANCH  branch to push results to  (created here from TCC_BRANCH)
#   TCC_MAX_HOURS       watchdog: terminate the pod after this many hours no matter what
#   GH_TOKEN            GitHub token with contents:write on the repo (push only; never written to disk)
#   HF_TOKEN            HuggingFace token for the gated models (Llama-3.2, Gemma-2)
#   RUNPOD_API_KEY      so the pod can terminate itself; RUNPOD_POD_ID is set by RunPod
#
# Durability: every 10 minutes a heartbeat commits pod_status/<job>/ (console log, pip freeze)
# and data/trackC/ (where --snapshot copies finished runs) to TCC_RESULTS_BRANCH and pushes.
# At the end the final snapshot is committed, pod_status/ is removed from the tree (its console
# log is kept under data/trackC/logs/), the push is retried for up to an hour, and the pod
# terminates itself. The watchdog terminates it regardless after TCC_MAX_HOURS.
set -uo pipefail

JOB="${TCC_JOB:?set TCC_JOB}"
CODE_BRANCH="${TCC_BRANCH:?set TCC_BRANCH}"
RESULTS_BRANCH="${TCC_RESULTS_BRANCH:?set TCC_RESULTS_BRANCH}"
MAX_HOURS="${TCC_MAX_HOURS:-2}"
REPO_DIR="${TCC_DIR:-/workspace/tcc}"
STATUS_DIR="pod_status/$JOB"
T_START=$(date +%s)

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*"; }

terminate_pod() {
  log "terminating pod ${RUNPOD_POD_ID:-?} (uptime $(( ($(date +%s) - T_START) / 60 )) min)"
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    curl -sS -X POST https://api.runpod.io/graphql \
      -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" >/dev/null 2>&1 || true
    sleep 20
    command -v runpodctl >/dev/null 2>&1 && runpodctl remove pod "$RUNPOD_POD_ID" >/dev/null 2>&1 || true
  fi
}

# --- watchdog backstop: terminate after MAX_HOURS whatever happens -------------------------
( sleep "$(python3 -c "print(int(float('$MAX_HOURS')*3600))")"; echo "[$(ts)] WATCHDOG: $MAX_HOURS h elapsed"; \
  terminate_pod ) &
WATCHDOG_PID=$!

cd "$REPO_DIR" || { log "no repo at $REPO_DIR"; terminate_pod; exit 1; }
mkdir -p "$STATUS_DIR"
exec > >(tee -a "$STATUS_DIR/console.log") 2>&1
log "=== job=$JOB code=$CODE_BRANCH results=$RESULTS_BRANCH max_hours=$MAX_HOURS pod=${RUNPOD_POD_ID:-?} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true

# --- git identity + credential (token stays in the environment) ----------------------------
git config user.name "tcc-pod"
git config user.email "tcc-pod@users.noreply.github.com"
git config credential.helper '!f() { echo "username=x-access-token"; echo "password=${GH_TOKEN}"; }; f'
git checkout -q -B "$RESULTS_BRANCH"

push_status() {                      # commit whatever is new under pod_status/ and data/trackC/, push
  local msg="$1"
  git add -A pod_status data/trackC 2>/dev/null
  git diff --cached --quiet && return 0
  git commit -q -m "$msg" || return 1
  git push -q -u origin "$RESULTS_BRANCH" 2>&1 | tail -2
}

# --- heartbeat: every 10 min -----------------------------------------------------------------
( while true; do sleep 600; push_status "pod $JOB: heartbeat $(ts)" >/dev/null 2>&1 || true; done ) &
HEARTBEAT_PID=$!

# --- bootstrap (venv + CUDA torch + deps; no model pre-download) -----------------------------
log "bootstrap"
PREDOWNLOAD=0 DIR="$REPO_DIR" bash deploy/bootstrap.sh || { log "BOOTSTRAP FAILED"; }
. .venv/bin/activate
python -c "import hf_transfer" 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0
python -m pip freeze > "$STATUS_DIR/pip-freeze.txt"
python -m tests.test_scoring && python -m tests.test_corpus && python -m tests.test_snapshot \
  && python -m experiments.e6_scale --selftest && python -m experiments.e4_scale --selftest \
  && python -m experiments.e5_scale --selftest || log "SELFTESTS FAILED (continuing so the log is pushed)"
push_status "pod $JOB: bootstrap done $(ts)"

mkdir -p results
run_one() {                           # run_one <label> <python -m args...>  (console to results/<label>.log too)
  local label="$1"; shift
  log ">>> $label :: $*"
  # direct redirect, not tee: the --snapshot copy runs at python exit and a tee pipe can lag
  # behind it, which truncated the console copies of the 2026-11-04 e4e5 runs (JSONs unaffected)
  python "$@" > "results/$label.console.log" 2>&1
  local rc=$?
  cat "results/$label.console.log"
  log "<<< $label exit=$rc"
}

run_job() {
  case "$JOB" in
    diag)
      # why did a gated model fail to load? identity of the HF token and the exact error (no values printed)
      log "HF_TOKEN present: $([ -n "${HF_TOKEN:-}" ] && echo yes len=${#HF_TOKEN} || echo NO)"
      log "HF env: $(env | grep -oE '^(HF_[A-Z_]*|HUGGINGFACE[A-Z_]*|TRANSFORMERS_OFFLINE)=' | tr '\n' ' ')"
      python -m pip show huggingface_hub transformers tokenizers 2>/dev/null | grep -E "^(Name|Version)"
      for url in https://huggingface.co/api/whoami-v2 https://huggingface.co/api/models/meta-llama/Llama-3.2-1B \
                 https://huggingface.co/api/models/google/gemma-2-2b https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json; do
        log "curl (auth)   $url -> HTTP $(curl -sS -o /tmp/r.json -w '%{http_code}' -H "Authorization: Bearer ${HF_TOKEN:-}" "$url"); $(python3 -c "
import json,sys
try: d=json.load(open('/tmp/r.json'))
except Exception as e: print('non-json body'); sys.exit()
if 'auth' in d: a=d['auth'].get('accessToken',{}); print('type',d.get('type'),'role',a.get('role'),'fineGrained',json.dumps(a.get('fineGrained',{}))[:300])
else: print({k:d[k] for k in ('error','gated','private','id') if k in d})
")"
        log "curl (noauth) $url -> HTTP $(curl -sS -o /dev/null -w '%{http_code}' "$url")"
      done
      python - <<'PY'
import traceback, os
from huggingface_hub import whoami, hf_hub_download
try: print("whoami:", {k: v for k, v in whoami().items() if k in ("type", "name")})
except Exception: traceback.print_exc()
for repo in ("meta-llama/Llama-3.2-1B", "google/gemma-2-2b", "Qwen/Qwen2.5-0.5B"):
    try: print("hf_hub_download OK:", repo, hf_hub_download(repo, "config.json"))
    except Exception: traceback.print_exc()
from transformers import AutoTokenizer
try: print("AutoTokenizer OK:", AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B").__class__.__name__)
except Exception: traceback.print_exc()
PY
      ;;
    smoke)
      # bootstrap check: all five tokenizer-family models (two gated), bf16, snapshot, push
      run_one smoke_e6B -m experiments.e6_scale --part B --trials 2 --cross-hops 1 \
        --device cuda --prefix results/smoke_e6B --snapshot trackC
      ;;
    p1)
      # plan item 2.1: E6 at power (3 seeds x 20 trials Part A; seeding sweep x 30 trials Part B)
      deploy/run_trackC_p1.sh
      ;;
    followup)
      # two follow-ups raised by the 2.1 results (2026-11-04):
      #  (a) a second 7B BASE family in serial passage, to separate scale from family: the only model that
      #      locked at 30-50% in 2.1 was also the only Qwen base model. Same protocol as Part A of
      #      run_trackC_p1.sh (hops 6, host0-reps 16, T 96, T=1.0, 20 trials x seeds 7 8 9).
      #  (b) the firebreak matrix at 1 HOP, the only configuration where a same-tokenizer advantage has
      #      ever been measured (2026-07-02, 8 trials), at the same n and seeding sweep as the 3-hop run.
      for s in ${SEEDS:-7 8 9}; do
        run_one "e6A_hiN_pythia69b_s$s" -m experiments.e6_scale --part A --serial-models EleutherAI/pythia-6.9b \
          --seed "$s" --trials 20 --device cuda --prefix "results/e6A_hiN_pythia69b_s$s" --snapshot trackC
      done
      for reps in ${REPS_SWEEP:-8 16 32}; do
        run_one "e6B_1hop_reps$reps" -m experiments.e6_scale --part B --cross-hops 1 --host0-reps "$reps" \
          --trials 30 --device cuda --prefix "results/e6B_1hop_reps$reps" --snapshot trackC
      done
      ;;
    e4e5)
      # plan items 2.2 and 2.3 on both instruct families; new prefixes so the 2026-07-02 files stay.
      # Phi-3.5-mini-instruct is an UNGATED supplementary second family, added 2026-11-03 because the
      # HF token could not read gated repos (Llama fails at load and is logged; rerun when fixed).
      for m in ${E4E5_MODELS:-Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.2-3B-Instruct microsoft/Phi-3.5-mini-instruct}; do
        tag=$(echo "$m" | sed 's#.*/##; s/Qwen2.5-7B-Instruct/qwen7b-instruct/; s/Llama-3.2-3B-Instruct/llama3b-instruct/; s/Phi-3.5-mini-instruct/phi35mini-instruct/')
        run_one "e4_scale_v2_$tag" -m experiments.e4_scale --model "$m" --device cuda \
          --prefix "results/e4_scale_v2_$tag" --snapshot trackC
        run_one "e5_scale_v2_$tag" -m experiments.e5_scale --model "$m" --context-tokens 4000 --device cuda \
          --prefix "results/e5_scale_v2_$tag" --snapshot trackC
      done
      ;;
    *) log "unknown TCC_JOB=$JOB" ;;
  esac
}

run_job
log "job body finished after $(( ($(date +%s) - T_START) / 60 )) min"

# --- final commit: keep the console log with the data, drop the transient status dir ---------
kill "$HEARTBEAT_PID" 2>/dev/null || true
mkdir -p data/trackC/logs
cp "$STATUS_DIR/console.log" "data/trackC/logs/pod_${JOB}_console.log" 2>/dev/null || true
cp "$STATUS_DIR/pip-freeze.txt" "data/trackC/logs/pod_${JOB}_pip-freeze.txt" 2>/dev/null || true
git add -A data/trackC
git rm -r -q --cached pod_status 2>/dev/null || true
git commit -q -m "pod $JOB: final data/trackC snapshot $(ts)" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  if git push -q -u origin "$RESULTS_BRANCH"; then log "pushed $RESULTS_BRANCH"; break; fi
  log "push failed (attempt $i), retrying in 5 min"; sleep 300
done
log "=== done; total $(( ($(date +%s) - T_START) / 60 )) min ==="
kill "$WATCHDOG_PID" 2>/dev/null || true
terminate_pod
