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
| `trackC/smoke_e6B.{json,png,console.log}` | **smoke test only, not a result**: E6-B at 1 hop, 2 trials, from the first unattended RunPod pod (2026-11-03, `deploy/runpod_job.sh`); the two gated families were skipped (see `logs/pod_diag_console.log`) |
| `trackC/logs/pod_*_console.log`, `pod_*_pip-freeze.txt` | console and environment of each unattended pod job (`deploy/runpod_job.sh`); `pod_diag_console.log` is the second diag run (token can read gated repos); the first, which recorded the `canReadGatedRepos: false` failure, is in git history at commit `0d143a2` |

## Still missing (raw data lost unless recovered from the dev Mac / Azure disk)
Track A: `e1_1_gpt2`, `e1_2_gpt2`, `e1_5_gpt2`, `e1_3`, `e2_gpt2`, `e4_trust`, `e5_hijack`,
`e6_worm`, `e7_detector`, `e8_substitution`. Track B: `e1_3_trackb_t4`, `e1_3_llama*`, `e1_4`,
`e2_gpt2-xl`, `e2_EleutherAI_pythia-*`, `e2_Qwen_Qwen2.5-7B`, `e8_capstone`. If any survive,
commit them here unmodified; otherwise their log tables stay marked "from run log, raw data lost."
