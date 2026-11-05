# Handoff: local reruns and data recovery on the MacBook

You are an agent running on the author's MacBook. This document brings you up to speed on
where this repository stands and tells you exactly what to do, in order. Read it fully
before running anything. Everything you need to know is either here or in the files it
names; do not rely on the paper's older claims.

## 1. Where things stand

- The repository tests whether a toy-model story about "context contagion" (repeated spans
  in an LLM's context getting copied back out, carried by induction heads) holds on real
  transformers. See `README.md` for the per-claim status table.
- On 2026-11-03 a critical review found several problems. The work plan that came out of it
  is `review_remediation_plan.md`; read its **Progress** section. Tier 0 (code fixes,
  re-analyses of committed data, writing) and the pre-run code for Tier 2 are **done**,
  on branch `claude/fervent-ritchie-l5m9uw`, which is open as
  [pull request #1](https://github.com/shumow/transformer-context-contagion/pull/1) into `main`.
- What has **not** happened: any experiment has been rerun. Three experiments currently have
  no valid numbers (E7 detector, E8-lite substitution, E4-scale), and most of the raw result
  files from the original runs were never committed (`data/README.md`, "Still missing").
- Your job is **Tier 1** of the plan plus **data recovery**: the things that need only this
  machine. Do not attempt Tier 2 (it needs a GPU pod).

## 2. Ground rules

1. **Branch workflow.** Start from the parent branch and work on a sub-branch:
   ```bash
   git fetch origin
   git checkout claude/fervent-ritchie-l5m9uw && git pull --ff-only
   git checkout -b claude/fervent-ritchie-l5m9uw-iter6-local-reruns
   ```
   Commit as you go, one commit per task below. When everything is done, push the
   sub-branch, merge it into the parent with `git merge --no-ff`, and push the parent. The
   open PR picks it up automatically. **Never push to `main`. Never force-push.**
2. **Raw data is sacred.** Commit result files under `data/<track>/` exactly as the script
   wrote them. Never edit a JSON by hand. Never modify or delete the `*_transcribed.json`
   files; if you recover the raw file a transcription stood in for, commit the raw file
   beside it and note in `data/README.md` that the transcription is now superseded.
3. **Report what you get, not what was hoped for.** Several of these reruns may weaken or
   remove a claim. That is a valid result. Write it in `results_log.md` plainly, with the
   numbers and their intervals, and update the README status table to match. Do not change
   experiment defaults, seeds or thresholds to improve a result; if you change any argument
   from the command given below, say so and why in the log entry.
4. **Payloads stay benign.** Read `SECURITY.md`. Nothing here needs a harmful string.
5. **This machine's limits are known.** 8 GB unified memory. GPT-2-small (124M) and
   Pythia-160M run fine; gpt2-medium and larger thrash swap and must not be attempted
   (`results_log.md`, "E1.3-lite"). TransformerLens on the MPS backend may be silently
   wrong on recent torch; every interpretability experiment below already defaults to CPU,
   keep it that way. Generation-only experiments may use MPS.
6. **Every run gets `--snapshot trackA`.** That flag copies the run's outputs into
   `data/trackA/` (git-tracked) when the script exits. Without it the output lands in the
   gitignored `results/` and is lost again.

## 3. Setup

```bash
cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch: the default wheel is fine for CPU/MPS
python -m tests.test_scoring && python -m tests.test_corpus && python -m tests.test_snapshot
python -m experiments.e7_detector --selftest
python -m experiments.e4_scale --selftest
```
All of those must pass before you run a model. If `transformer_lens` fails to import,
check that `transformers` resolved to a 4.x version (the requirement pins `<5`).

## 4. Tasks, in order

### Task 1 — recover lost raw results (no compute; do this first)

The original Track A runs were made on this machine in June 2026 and their JSON/PNG files
were written to a gitignored `results/` directory. They may still exist in an old clone, a
backup, or Downloads. Search:
```bash
find ~ -type f \( -name 'e1_1_gpt2.json' -o -name 'e1_2_gpt2.json' -o -name 'e1_5_gpt2.json' \
  -o -name 'e1_3.json' -o -name 'e2_gpt2.json' -o -name 'e4_trust.json' -o -name 'e5_hijack.json' \
  -o -name 'e6_worm.json' -o -name 'e7_detector.json' -o -name 'e8_substitution.json' \
  -o -name 'e1_3_trackb_t4.json' -o -name 'e1_4.json' -o -name 'e2_gpt2-xl.json' \
  -o -name 'e8_capstone.json' \) 2>/dev/null
```
Also look for anything `scp`'d down from the Azure VM or the RunPod pod (`e1_3_llama*`,
`e2_EleutherAI_*`, `e2_Qwen_*`). For every file found: copy it **unmodified** to
`data/trackA/` (GPT-2 / Pythia-160M runs) or `data/trackB/` (T4 / L40S runs), with its PNGs
if present; add a row to the table in `data/README.md` saying where it was found; and remove
that name from the "Still missing" list. If a recovered file corresponds to a
`*_transcribed.json`, keep both and mark the transcription superseded. Commit:
`Recover Track A/B raw results from the dev machine`.

If nothing is found, say so in `data/README.md` ("searched <date>, not found") and move on.

### Task 2 — E7 detector rerun (CPU, GPT-2-small; ~10–30 min)

```bash
python -m experiments.e7_detector --snapshot trackA
```
This replaces the three-negative, no-baseline run. It reports, at each k, the AUC of the
induction-head score **and** of a model-free n-gram-repeat count, with bootstrap CIs, plus
the false-positive rate on eight legitimately repetitive texts at the zero-FP-on-clean
threshold. Write a new entry in `results_log.md` headed
`## E7 rerun — detector vs n-gram baseline, 32 negatives (<date>, local CPU)` with the
table the script prints and a verdict on this question: *does the induction score beat the
n-gram count?* If the AUCs are indistinguishable, the honest sentence is "the detector is an
n-gram repeat detector; the induction machinery explains why repetition is dangerous but is
not a better alarm." Report the benign-repetitive FPR as a number. Then edit the
`> **Status (2026-11-03): superseded pending rerun.**` note at the old E7 entry to point at
the new one, and update the E7 row of the README status table. Commit.

### Task 3 — E8-lite rerun with the corrected control (CPU, GPT-2-small; ~10–30 min)

```bash
python -m experiments.e8_substitution --snapshot trackA
```
The old run's "random" control was the eight lowest-induction heads. The script now reports
the induction ablation against both a true seeded random set and the lowest-induction set.
New log entry `## E8-lite rerun — true random control (<date>, local CPU)` with the table.
The question: *is the induction-ablation drop still much larger than the random-ablation
drop?* If the random drop is comparable, say the "substitution is induction-carried" claim
is weakened and by how much. Update the old entry's status note and the README row. Commit.

### Task 4 — E1.1 rerun so the raw curves exist (GPT-2-small; MPS or CPU; ~20–40 min)

```bash
python -m experiments.e1_condensation_knee --model gpt2 \
    --lengths 1 2 3 5 --reps 1 2 4 8 16 32 --payloads 4 --trials 10 --max-new 48 \
    --snapshot trackA
python -m experiments.analysis_knee_floor data/trackA/e1_1_gpt2.json \
    --out data/trackA/analysis_knee_floor_gpt2.json
```
This is the same command as the original E1.1 (sampling, so numbers will differ a little
from the log's). The point is to have the per-N curves committed, and to report the
**direct** one-shot copy rate P(N=2) with its Wilson interval per length, which the analysis
prints. Log entry `## E1.1 rerun — curves committed, direct one-shot rate (<date>)`: the knee
table, and a second table of P(N=1), P(N=2) per p. Commit.

### Task 5 — E6 worm rerun with per-trial trajectories (GPT-2 + Pythia-160M; ~30–60 min)

```bash
python -m experiments.e6_worm --trials 8 --snapshot trackA
python -m experiments.analysis_e6_r0 data/trackA/e6_worm.json --out data/trackA/analysis_e6_r0_gpt2.json
```
The script now stores every trial's trajectory, so the R0 analysis can give per-trial
medians and bootstrap CIs on survival and on the per-hop ratio, which the committed Track C
data could not. Log entry `## E6 rerun — per-trial R0 and survival CIs (<date>)` with the
analysis table. The question: *how many of the "endemic" p=1 trials are fully locked loops
versus genuinely fluctuating loads?* The analysis prints a `fully_locked` count. Commit.

### Task 6 — optional, if time remains (same machine, same pattern)

In this order, each with `--snapshot trackA` and a short log entry:
- `python -m experiments.e1_2_cue --model gpt2 --lengths 1 2 3 5 8 --reps 16 32 --payloads 6 --trials 12`
- `python -m experiments.e1_5_context --model gpt2 --lengths 1 3 5 --reps 1 2 4 8 16 32 --gaps 0 8 16 32 64 128 --payloads 4 --trials 10`
- `python -m experiments.e2_induction --model gpt2 --device cpu --lengths 1 3 5 --payloads 6 --k-heads 8`
- `python -m experiments.e5_hijack`
- `python -m experiments.e4_trust`
These only recreate raw data for entries whose files were lost; no claim changes.

### Task 7 — rebuild the PDF if TeX is installed

```bash
which pdflatex latexmk
cd latex && latexmk -pdf transformer_context_contagion.tex   # or pdflatex twice
```
If it builds, commit the new PDF and delete the two sentences saying the PDF predates the
revision (in the paper's revision note after `\maketitle`, and in `README.md` under
"Status"). If TeX is absent, leave both sentences alone.

### Task 8 — close out

- Update the **Progress** table in `review_remediation_plan.md`: Tier 1 row → done, with
  what ran and what did not.
- `git push -u origin <sub-branch>`; then
  `git checkout claude/fervent-ritchie-l5m9uw && git merge --no-ff <sub-branch> && git push`.
- Write a short summary for the author: which tasks ran, the headline number from each
  (with its interval), which claims changed, what failed, and how long each run took.

## 5. How to write a results_log entry

Follow the existing entries. Each new one has: a dated heading; the exact command; a table
of the numbers as the script printed them (not rounded further); one paragraph answering the
question the task poses; a caveats line (n, seeds, device, runtime). Where a rerun replaces an
older entry, do not delete the old entry; add the status pointer at its top.

## 6. If something breaks

- A model download fails: `HF_HUB_OFFLINE=1` makes `from_pretrained` use the local cache;
  gpt2, gpt2's tokenizer and Pythia-160M were cached on this machine in June.
- `transformer_lens` warns about MPS: that is expected; the interp scripts are on CPU.
- A run is too slow: reduce `--trials` or `--payloads`, say so in the log entry, and keep
  the reduced command there.
- Anything about the repository you cannot resolve from the files: stop, write what you
  found and where you stopped into the summary for the author, and do not guess.
