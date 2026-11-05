# Committed data snapshots

`results/` is **gitignored** (outputs are meant to be reproducible from `experiments/*.py`).
That is fine for runs anyone can cheaply regenerate locally, but it is *not* fine for at-scale
runs produced on ephemeral cloud GPUs (RunPod/Azure): when the pod is torn down, the only record
is whatever was hand-transcribed into `results_log.md`.

This directory is the **durable** home for the raw JSON (and key PNGs) of runs that are expensive
or impossible to reproduce locally. Unlike `results/`, it is tracked by git.

## Convention
- One subdirectory per track/run set: `data/trackA/`, `data/trackB/`, `data/trackC/`.
- Commit the exact `results/<exp>.json` the log tables were transcribed from, unmodified.
- Note the pod/GPU, date, and the exact command in the accompanying entry in `results_log.md`.
- Pod run scripts (`deploy/run_*.sh`) should copy `results/*.json` here as their last step, so
  this never again depends on a manual pull before teardown.
- A file whose name ends in `_transcribed.json` (or carries a `_provenance` field) is **not raw
  output**: it was typed back in from a table in `results_log.md` or a run log because the raw
  JSON was lost. Treat its numbers as the log's numbers, with the log's rounding.
- `analysis_*.json` files are outputs of `experiments/analysis_*.py` run on the files beside them.

## What is here (2026-11-03)
| file | status |
|---|---|
| `trackC/e4_scale.json`, `e5_scale.json`, `e6_scale.json` (+ PNGs, `logs/`) | raw, from the RunPod L40S, 2026-07-02 (`e6_scale.json` holds the **second** Part-B run only: 1 hop, 32 seed copies, 8 trials) |
| `trackC/e6_scale_partB_3hop_reps16.json` | transcribed from `logs/e6.log`: the **first** Part-B run (3 hops, 16 seed copies, 5 trials), overwritten on the pod |
| `trackA/e6_worm_transcribed.json` | transcribed from `results_log.md` (E6 on GPT-2 / Pythia-160M); raw file not recovered |
| `trackB/e1_3_e1_4_knees_transcribed.json` | transcribed from `results_log.md` (E1.3 / E1.4 knee tables); raw files on the Azure VM disk were not recovered |
| `trackB/analysis_knee_floor.json`, `trackC/analysis_e6_firebreak.json`, `trackC/analysis_e6_r0.json` | analysis outputs, see `results_log.md` "Analyses of committed data" |
| `trackC/e6A_hiN_s{7,8,9}.{json,log,png}`, `e6B_reps{8,16,32}.{json,log,png}` | raw, RunPod L40S 2026-11-03/04, `deploy/run_trackC_p1.sh` defaults (plan 2.1); per-trial trajectories under `runs` / `cross_trials`; `analysis_e6_r0_pooled_hiN.json`, `analysis_e6_firebreak_sweep.json` are their analyses |
| `trackC/e5_scale_v2_rawpad_v1_*.{json,png,console.log}` | raw, RunPod L40S 2026-11-05; the **first** `rawpad` control (commit `ff07531`, prose filler from the dump's own sentence bank). Superseded by the `e5_scale_v2_rawpad_*` files of the stricter control in `experiments/e5_scale.py` (disjoint prose bank + token pad, `POD_FOLLOWUP3_BRIEF.md`); kept unmodified |
| `trackC/e4_scale_v2_*.{json,png}`, `e5_scale_v2_*.{json,png}` | raw, RunPod L40S 2026-11-04 (plan 2.2, 2.3), three instruct models; the matching `*.console.log` copies are **truncated** (tee/snapshot race in the first version of `deploy/runpod_job.sh`, since fixed) — the JSONs are complete and are the record |
| `trackC/smoke_e6B.{json,png,console.log}` | **smoke test only, not a result**: E6-B at 1 hop, 2 trials, from the first unattended RunPod pod (2026-11-03, `deploy/runpod_job.sh`); the two gated families were skipped (see `logs/pod_diag_console.log`) |
| `trackC/logs/pod_*_console.log`, `pod_*_pip-freeze.txt` | console and environment of each unattended pod job (`deploy/runpod_job.sh`); `pod_diag_console.log` is the second diag run (token can read gated repos); the first, which recorded the `canReadGatedRepos: false` failure, is in git history at commit `0d143a2` |

## Recovered from the dev Mac (2026-11-03)
The original Track A/B runs' JSON/PNG/log files were found still present in the gitignored
`results/` directory on the author's MacBook and copied here unmodified (`cp -p`, byte-identical).
| file | status |
|---|---|
| `trackA/e1_1_gpt2.json`, `e1_2_gpt2.json`, `e1_3.json`, `e1_5_gpt2.json`, `e2_gpt2.json`, `e4_trust.json`, `e5_hijack.json`, `e6_worm.json`, `e7_detector.json`, `e8_substitution.json` (+ their PNGs and `*_run.log`) | **raw**, recovered 2026-11-03 from `results/`; original local GPT-2-small / Pythia-160m runs, 2026-06-25–29 |
| `trackA/e6_worm.json` | raw E6 worm; **supersedes** `trackA/e6_worm_transcribed.json` (transcription kept per convention, now superseded by this raw file) |
| `trackA/e1_greedy_gpt2.json` (+ `e1_greedy_*.png`, `e1_knee.png`) | raw GPT-2 greedy-decoding companion run to E1.1; recovered 2026-11-03; was not on the missing list below but preserved as raw output |
| `trackB/e1_3_trackb_t4.json` (+ PNG), `e2_gpt2-xl.json`, `e2_EleutherAI_pythia-2.8b.json`, `e2_EleutherAI_pythia-6.9b.json`, `e2_Qwen_Qwen2.5-7B.json`, `e8_capstone.json` | **raw**, recovered 2026-11-03 from `results/`; T4/L40S cloud runs scp'd to the Mac, 2026-06-30–07-02. Verified 2026-11-04 against the log tables: every E2 cell and every E1.3 knee agree (one 2.25 vs 2.3 rounding). |
| `trackB/e1_3_trackb_t4.json` | raw E1.3 scale-knee run for the **ten ungated models only**; **partially supersedes** `trackB/e1_3_e1_4_knees_transcribed.json` (those ten rows — the Llama-3.2 rows and all of E1.4 remain transcription-only, so the file is kept). Stores knees only, not per-N curves (the script did not retain them until 2026-11-03). |
| `trackB/e1_3_llama_t4.json` | **NOT raw support for the Llama rows.** This is the *failed* Llama attempt: both models errored with a gated-repo 401 (`HF_HUB_OFFLINE` was not set). The successful offline rerun that produced the logged Llama-3.2-1B/3B knees was a separate file that was not recovered. Kept as a record of the failure; the Llama knees stay "from run log, raw data lost". |
| `trackB/e2_EleutherAI_pythia-1.4b.json` | **raw for the first (T4, fixed 8-head) Pythia-1.4B E2 run only** (`k_heads=8`, `k_frac=null`, p=3 N=2: −80% / random −2%, matching the 2026-07-01 entry). The **C2-tidy rerun at `--k-frac 0.05` (19/384 heads, −88%)** that the E2-at-scale table and paper `tab:e2scale` report overwrote this filename on the pod and was not recovered; that row stays "from run log, raw data lost". |

**Note on `trackA/e7_detector.json`:** the E7 rerun (2026-11-03, Task 2) writes to this same
filename, so `trackA/e7_detector.json` (+ `.png`) now holds the **32-negative rerun**, not the
recovered June run. The recovered June 3-negative run is preserved beside it as
`trackA/e7_detector_2026-06-26_3neg.json` (+ `.png`). See `results_log.md`, "E7 rerun".

**Note on `trackA/e8_substitution.json`:** likewise the E8-lite rerun (2026-11-03, Task 3) writes
to this filename, so `trackA/e8_substitution.json` (+ `.png`) now holds the **true-random-control
rerun**. The recovered June run is preserved as `trackA/e8_substitution_2026-06-26.json` (+ `.png`).
See `results_log.md`, "E8-lite rerun".

**Note on `trackA/e1_1_gpt2.json`:** the E1.1 rerun (2026-11-03, Task 4) writes to this filename,
so `trackA/e1_1_gpt2.json` (+ `_knee_`/`_Nstar_` PNGs) now holds the **2026-11-03 sampled rerun**
whose per-N curves are committed for the first time; `trackA/analysis_knee_floor_gpt2.json` is its
knee-floor analysis. The recovered June run is preserved as `trackA/e1_1_gpt2_2026-06-25.json`
(+ dated PNGs). See `results_log.md`, "E1.1 rerun".

**Note on `trackA/e6_worm.json`:** the E6 rerun (2026-11-03, Task 5) writes to this filename, so
`trackA/e6_worm.json` (+ `_serial`/`_crosstok` PNGs) now holds the **trials=8 rerun with per-trial
trajectories** (`runs`); `trackA/analysis_e6_r0_gpt2.json` is its R0/survival-CI analysis. The
recovered June run is preserved as `trackA/e6_worm_2026-06-26.json` (+ dated PNGs); this raw run
also still supersedes `trackA/e6_worm_transcribed.json` (kept). See `results_log.md`, "E6 rerun".

## Still missing (raw data lost unless recovered from a cloud disk)
Track A: none — all recovered 2026-11-03 (see above).
Track B, three files (searched the dev Mac 2026-11-03, not found):
- `e1_4.json` — the instruct-model knee sweep (Azure T4 VM disk, if it still exists);
- the successful Llama-3.2 E1.3 run (Azure T4 VM disk) — the recovered `e1_3_llama_t4.json` is the failed attempt;
- `e2_EleutherAI_pythia-1.4b.json` at `--k-frac 0.05` (RunPod pod `/workspace`, if the volume persisted) — the recovered file is the earlier 8-head run.
Their numbers remain in `trackB/e1_3_e1_4_knees_transcribed.json` and the log tables, marked
"from run log, raw data lost". If any survives, fetch with `deploy/fetch_results.sh` and commit
unmodified.
