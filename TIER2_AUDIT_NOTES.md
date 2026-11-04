# Tier 2 audit notes (RunPod runs of 2026-11-03/04)

For the advisor / auditor reviewing the Tier 2 work of `review_remediation_plan.md`. This file
says what was run, by what, where every number came from, what deviated from the plan or the
instructions and why, and how to re-derive every reported figure from committed files. It does
not restate findings beyond what is needed to audit them; the record is `results_log.md`
("E6 at power", "E4-scale v2", "E5-scale v2", and the follow-up entries).

## 1. Scope and provenance

- **Work items:** plan 2.1 (E6 at power), 2.2 (E4-scale v2), 2.3 (E5-scale with controls).
  2.4 was evaluated against its own trigger condition and not run (§6).
- **Who / where:** a Claude Code cloud session (no GPU, no Hugging Face access) orchestrating
  RunPod pods through the GraphQL API only. The pods ran unattended and pushed their own results.
  A separate agent on the author's laptop ran Tier 1 and the data recovery concurrently
  (iterations 6 and 8); the two streams merged in iteration 7's merge commit `f59fd17`.
- **Branches:** work on `claude/fervent-ritchie-l5m9uw-iter7-tier2-pods`; each pod pushed to its
  own branch off that (`…-iter7-pod-{smoke,diag,diag2,p1,e4e5}`); everything merged `--no-ff`
  into the parent `claude/fervent-ritchie-l5m9uw` (open as PR #1). Nothing was pushed to `main`,
  nothing was force-pushed. Follow-ups (§8) are on `…-iter9-followups`.
- **Git identity of pod commits:** `tcc-pod <tcc-pod@users.noreply.github.com>`, pushed with a
  fine-grained PAT (contents:write on this repo only). The PAT, the RunPod key and the HF token
  were passed to pods as environment variables and never written to any file or log; the diag
  logs print only their presence and length.

## 2. Pod jobs, in order

| job | pod id | cloud | created → gone (UTC) | wall | final commit | outcome |
|---|---|---|---|---|---|---|
| smoke | 21xtjk1pd62dnd | secure L40S | 11-03 22:59 → 23:05 | 6 min | `ea5196d` | bootstrap, selftests, push, self-terminate all OK; **Llama/Gemma failed to load** |
| diag | d3g1izm6p37qqc | secure L40S | 23:06 → 23:10 | 4 min | `0d143a2` | HF token: `canReadGatedRepos: false` (403 on gated file download) |
| p1 (first) | z1h9wm58miz1gm | secure L40S | 23:10 → 23:26 | 16 min | — | **never got a container** (runtime null); terminated; ≈$0.09 |
| diag2 | 62mis5c4p8ryka | secure L40S | 23:23 → 23:27 | 4 min | `8d5f57a` | after the author changed the token's permission: gated downloads OK |
| p1 | gkoh823pupxszu | secure L40S | 23:26 → 11-04 06:49 | 7 h 23 min | `e7fbd75` | 6/6 runs OK |
| e4e5 | yln3iriq7jiilt | secure L40S | 06:47 → 07:30 | 43 min | `2f8c15f` | 6/6 runs OK |

Total ≈ 8.4 pod-hours at $1.09/h; RunPod balance $50.00 → $41.01 (**$8.99**). The community
cloud ($0.79/h) had no L40S stock at any attempt. Every pod terminated itself; the account had
zero pods after each job (checked by API).

Pod environment (identical for every job, from `deploy/bootstrap.sh` with `PREDOWNLOAD=0`):
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, Python 3.11, torch 2.5.1+cu121,
transformers 4.57.6, huggingface_hub 0.36.2, bf16. Full freeze: `requirements-lock.txt`
(= `data/trackC/logs/pod_diag_pip-freeze.txt`). All model-free tests and the three
`--selftest`s ran on every pod before its job (see each `logs/pod_*_console.log`).

## 3. Exact commands

All defaults; nothing in any experiment file or run script was changed to obtain a number.

```
# 2.1  (deploy/run_trackC_p1.sh, unmodified; its defaults:)
python -m experiments.e6_scale --part A --seed {7,8,9} --trials 20 --device cuda --prefix results/e6A_hiN_s$seed --snapshot trackC
python -m experiments.e6_scale --part B --host0-reps {8,16,32} --trials 30 --device cuda --prefix results/e6B_reps$reps --snapshot trackC
# 2.2
python -m experiments.e4_scale --model <M> --device cuda --prefix results/e4_scale_v2_<tag> --snapshot trackC
# 2.3
python -m experiments.e5_scale --model <M> --context-tokens 4000 --device cuda --prefix results/e5_scale_v2_<tag> --snapshot trackC
#   <M> ∈ Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.2-3B-Instruct, microsoft/Phi-3.5-mini-instruct
```

The wrapper is `deploy/runpod_job.sh` (job names `p1`, `e4e5`); the pod was created by
`deploy/runpod_pod.py create`. Both are in the repo and contain no secrets.

## 4. Where each number comes from

| reported in | derived from (committed) | how |
|---|---|---|
| 2.1 Part A survival table, Wilson CIs | `data/trackC/e6A_hiN_s{7,8,9}.json` → `serial[model][p].runs` | 60 trials pooled; `k = #(last-hop load ≥ 1)`, `tcc.scoring.wilson_interval(k, 60)` |
| 2.1 Part A R_geo medians, bootstrap CIs, per-seed survival | same files | `python -m experiments.analysis_e6_r0 --pool data/trackC/e6A_hiN_s7.json data/trackC/e6A_hiN_s8.json data/trackC/e6A_hiN_s9.json` → `data/trackC/analysis_e6_r0_pooled_hiN.json` |
| "survivors sit at T/p" | same files | median/max of last-hop load over trials with load ≥ 1 (one-off script in the session; trivially recomputable from `runs`) |
| 2.1 Part B matrices, ratio, permutation p, bootstrap CI, column means | `data/trackC/e6B_reps{8,16,32}.json` → `cross`, `cross_trials`, `cross_config` | `python -m experiments.analysis_e6_firebreak data/trackC/e6B_reps8.json data/trackC/e6B_reps16.json data/trackC/e6B_reps32.json` → `analysis_e6_firebreak_sweep.json` |
| 2.2 2×2 means, correlations, CIs, Wilson on P_probe | `data/trackC/e4_scale_v2_*.json` → `means`, `corr`, `corr_ci`, `rows`, `length_match` | fields are written by `e4_scale.py` itself; Wilson from `rows` |
| 2.3 placement / repetition / control cells, `control_ok` | `data/trackC/e5_scale_v2_*.json` → `placement`, `repetition`, `control`, `config` | fields written by `e5_scale.py`; per-trial occupancies under `trials` |

**Console logs.** The per-run `*.console.log` copies of the e4e5 runs are truncated (a `tee`
pipe lagged the `--snapshot` exit hook; fixed in `runpod_job.sh` by a direct redirect). The
pod-level `data/trackC/logs/pod_e4e5_console.log` is complete. Cross-check performed on
2026-11-04: every printed E4 correlation and interval (12/12) and every printed E5 k-row
(18/18) matches the JSON to the printed precision. The 2.1 per-run logs (`e6A_*.log`,
`e6B_*.log`) are complete.

## 5. Deviations from the plan or the instructions, with reasons

1. **Phi-3.5-mini-instruct added to 2.2 and 2.3** (not in the plan). Added while the HF token
   could not read gated repos, so that a second instruct family would exist even if Llama
   stayed blocked. The token was fixed before the pod ran, so Llama ran as planned and Phi is a
   third family. It is labeled "supplement" in the log; nothing was dropped to make room.
2. **Two pods briefly ran at once** (diag2 alongside the stuck first p1 pod, ≈6 min), on the
   author's explicit request to test the token immediately. The "one pod at a time" rule held
   otherwise and is enforced by `runpod_pod.py` unless `--allow-multiple` is passed.
3. **The first p1 pod was terminated** after 16 minutes with no container runtime; the diag2
   pod had started and finished on the same cloud in that window. Cost ≈ $0.09. Relaunched
   unchanged.
4. **Analysis scripts gained options**; no defaults or statistics changed: `analysis_e6_r0
   --pool` (concatenate per-trial runs across seed files), `analysis_e6_firebreak` bootstrap CI
   over `cross_trials`, row/column marginal means, and reading `cross_config` (the key
   `e6_scale.py` actually writes). All three re-reproduce the 2026-11-03 numbers on the old files.
5. **`requirements-lock.txt`** was created from the pod freeze (plan 0.8 asked for it from "the
   next working GPU environment").
6. **2.4 not run.** Its condition was "only if 2.1's seeding-strength curve still shows partial
   transmission." At 3 hops it shows none (§6), so by the plan's own rule the item lapses.
7. **The p = 1 column of Part A is reported but marked invalid** as transmission: a one-token
   payload decodes to a short common substring and text-level counting matches it by accident
   (documented in `tcc/payloads.py`); its 0.22–0.33 "survival" is the false-positive floor.
8. **Paper not touched** for C4–C6; the instructions named `results_log.md`, the README table
   and the plan's Progress section, and the paper rewrite is Tier 3.

## 6. Claim-by-claim: what the Tier 2 data changed

| claim | before (2026-07 / 11-03) | after (n, interval) | status |
|---|---|---|---|
| C6 "worm worse at scale" | Qwen2.5-7B base and instruct survive 20–60% of 5 trials | base 0.30–0.50 (Wilson lower bounds 0.20–0.38, n=60 per p); **instruct 0.08–0.13**; Pythia-2.8B 0.05–0.20; Llama-3.2-3B 0.00–0.12 | narrowed to one 7B base model; scale vs family confounded (follow-up §8a) |
| C6 R0 / epidemic | "R0 ≈ T/(p·k)" in abstract; re-analysis of means said no R>1 | per-trial R_geo medians ≤ 1.00 in every cell; every survivor's last-hop load = T/p | confirmed: locking probability, not reproduction number |
| C6 firebreak | 1-hop/32-copy run p=0.004 (n=8); 3-hop/16-copy run p=0.24 (n=5) | 3 hops, n=30: ratio 0.87 / 0.94 / 0.85 at 8/16/32 copies, bootstrap CIs all span 1, p = 0.59 / 0.59 / 0.65; host column (Qwen-0.5B) dominates | withdrawn as a multi-hop claim; 1-hop result untested at n (follow-up §8b) |
| C4 trust not usefulness (at scale) | corr(P,U)=+0.09, corr(P,T)=+0.71 on a flawed design, 6 facts | corr(P_probe,T) = +0.38 [+0.26,+0.50] / +0.15 [+0.06,+0.25] / +0.34 [+0.26,+0.42]; corr(P_probe,U) = −0.18 [−0.31,−0.04] / −0.04 / +0.07 (Qwen / Llama / Phi, 32 facts) | holds in direction on three models; smaller effect; Llama near ceiling at probe_k=2 |
| C5 bound at scale | occupancy 0.00, one model, one query, no control | 0.00–0.02 in 45 query cells (3 models × 3 queries × 5 k), max trial 0.095; raw-continuation control 0.59–1.00 from k=4 | null confirmed and now interpretable; new: chat template alone defuses |

## 7. Verification the auditor can repeat

```
python -m tests.test_scoring && python -m tests.test_corpus && python -m tests.test_snapshot
python -m experiments.analysis_e6_r0 --selftest && python -m experiments.analysis_e6_firebreak --selftest
python -m experiments.analysis_e6_r0 --pool data/trackC/e6A_hiN_s7.json data/trackC/e6A_hiN_s8.json data/trackC/e6A_hiN_s9.json
python -m experiments.analysis_e6_firebreak data/trackC/e6B_reps8.json data/trackC/e6B_reps16.json data/trackC/e6B_reps32.json
python - <<'PY'
import json; r=json.load(open("data/trackC/e4_scale_v2_qwen7b-instruct.json")); print(r["corr"], r["corr_ci"])
r=json.load(open("data/trackC/e5_scale_v2_qwen7b-instruct.json")); print(r["control_ok"], r["repetition"]["summarize"]["32"])
PY
```
Each JSON carries the configuration it was run with (`cross_config`, `config`, `k`, `probe_k`,
`seed`, `model`). Pod console logs list the image, GPU, driver, versions and selftest output.

## 8. Follow-ups raised by these results (in progress on `…-iter9-followups`)

- (a) **Pythia-6.9B serial passage**, 3 seeds × 20 trials, same protocol as Part A: separates
  "7B" from "Qwen" in the one surviving scale claim.
- (b) **1-hop firebreak sweep**, 8/16/32 copies × 30 trials: the only configuration in which a
  same-tokenizer advantage has been measured, previously at n=8.
- Not started: Llama-3.2-3B-Instruct in E4 v2 at `--probe-k 1` or sampled trials (ceiling);
  the paper's C4–C6 text (Tier 3).
