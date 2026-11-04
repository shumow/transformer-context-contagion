# Review remediation plan

A plan to act on the critical review of this repository (2026-11-03). The review's
conclusion: the engineering and the lab-notebook discipline are strong, but the headline
"condensation knee" is mostly a floor effect, several results rest on design artifacts or
thin samples, the related-work coverage misses the closest prior art, and the writing
restates findings in five places with drift. This document turns each finding into a
concrete, ordered work item with a definition of done.

Items are grouped by what they need: **Tier 0** needs no model runs; **Tier 1** needs a
CPU and small models; **Tier 2** needs a GPU box. Within a tier, order is by leverage.
Each item names the files it touches so the work can be checked against it.

---

## Tier 0 — code, analysis, and writing fixes (no model runs)

### 0.1 Fix the E8-lite control: it is not random
`experiments/e8_substitution.py:103` selects the eight **lowest-induction-score** heads as
the control but labels them "random heads" in the log, the plot, and the results entry.
Lowest-induction heads are likely unimportant in general, which biases the control toward
showing no effect.
- Draw a genuinely random matched set (excluding the induction set), seeded, the way
  `e2_induction.py:76-80` already does.
- Keep the lowest-induction set as a second, correctly labeled control.
- Mark the E8-lite table in `results_log.md` as superseded pending the Tier 1 rerun.
- **Done when:** script emits both controls under their true names; rerun lands in 1.2.

### 0.2 Fix the E4-scale design: the "non-repetitive" contexts repeat
`experiments/e4_scale.py:66-67` builds both non-repetitive cells as the **same filler
paragraph twice**, so the 2×2 is really 8× repetition vs 2× repetition. Also, the
poisonability metric is copy-rate of a raw continuation of "fact fact fact…", which is
1.0 for any repeated span by construction.
- Replace `FILLER` with a bank of ≥8 distinct paragraphs; non-repetitive contexts draw
  distinct paragraphs, length-matched to the repetitive cells in tokens.
- Add the poisonability measure the original plan specified: plant an E1-style OOD probe
  span (fixed `k_probe`, at the end) into **each** of the four contexts and measure
  whether it is reproduced. This separates "the context is itself a loop" from "the
  context makes the model more poisonable."
- Report both P metrics side by side; the correlation table uses the probe version.
- Raise the fact set from 6 to ≥30 (write them; keep them synthetic and benign).
- **Done when:** selftest checks no paragraph appears twice in a non-repetitive context;
  the rerun lands in 2.2.

### 0.3 Add the trivial baseline to E7, and enough negatives for an AUC to mean something
`experiments/e7_detector.py` reports AUC 1.0 on **three** clean documents. It also never
compares the induction-head score to a model-free "does any n-gram repeat" count, which
would score the same on this data.
- Add `ngram_repeat_score(ids, n)` (count of repeated n-grams per token) as a baseline
  column; report its AUC next to the induction score at every k.
- Expand negatives to ≥30 clean paragraphs (hard-coded benign prose, no external dataset
  dependency), and add a **benign-repetition** negative set (boilerplate, a license
  header, a code snippet, a repeated chorus) to report the false-positive rate honestly.
- Report AUC with a bootstrap CI.
- **Done when:** the entry states whether the induction signal beats the n-gram count,
  and the false-positive rate on benign repetition is a number.

### 0.4 Report both firebreak runs, not one
`data/trackC/logs/e6.log` contains two Part B runs. The first (script defaults: 3 hops,
16 seed copies, 5 trials) has pythia→pythia = 0.0, llama→llama = 0.0, gemma→gemma = 0.8.
Only the second (1 hop, 32 seed copies, 8 trials) is reported in `results_log.md` and the
paper. One hop is barely a transmission test.
- Add the 3-hop matrix to the E6-scale entry and to the paper's C6 paragraph.
- Compute diagonal vs off-diagonal means for both runs (from the committed JSON and log)
  with a permutation test on "diagonal is largest in its row."
- State the protocol change and the reason explicitly. The claim becomes "at 1 hop under
  heavy seeding the same-tokenizer diagonal is ~1.6× the off-diagonal; at 3 hops on
  0.5–2B models nothing reliably transmits either way."
- **Done when:** both matrices are in the log and paper, and the "partial firebreak"
  sentence cites the hop count and seeding it was measured at.

### 0.5 Reframe the knee: report the one-shot copy rate, and name the floor
For every model and every p ≥ 3 the interpolated knee lands in (1, 2). At N = 1 the cyclic
transition S[-1]→S[0] has never occurred in context, so copying it is structurally
impossible; at N = 2 it has occurred once. The "sharp, scale-flat knee" is the grid floor.
- Add to `tcc/scoring.py` a `one_shot_rate` summary: P(reproduce) at N = 2 directly, with
  its Wilson interval, and P at N = 1 as the chance floor.
- Add a short analysis script `experiments/analysis_knee_floor.py` that, from any E1 JSON,
  reports: one-shot rate per (model, p); whether N* is at the floor; and the p = 1, 2
  regime separately.
- In the paper and README, replace "condensation knee decreasing with length" with "the
  one-shot copy rate for novel spans of ≥ 3 tokens is ≥ 0.5 from 355M to 7B; single-token
  and bigram spans are the exception, driven by frequency priors and copy suppression."
- Keep the knee figure, but caption it as a floor and say so in the text.
- **Done when:** no sentence in README, ONEPAGER, or the paper calls the p ≥ 3 threshold a
  discovered phase transition or claims it is "flat with scale" as a finding.

### 0.6 Actually compute R0 and the critical length from the committed E6 data
R0 is in the plan and the abstract but is computed nowhere.
- Add `experiments/analysis_e6_r0.py`: from per-hop loads, fit the one-hop map
  k_out = f(k_in) near threshold and report R0 = dk_out/dk_in at k_in → small, per
  (model, p), with a bootstrap over trials. Report critical length as the smallest p with
  R0 < 1 (or "none ≤ 8").
- Needs per-trial data; the committed `e6_scale.json` only has means. Modify
  `e6_worm.py` / `e6_scale.py` to also store per-trial `ks`, so the Tier 2 rerun (2.1)
  supports this. Compute what can be computed from means now and label it as such.
- **Done when:** the results log has an R0 table with intervals, or states the data is
  insufficient and which rerun fixes it.

### 0.7 Related work the paper is missing
Add to `latex/transformer_context_contagion.tex` §3 and the bibliography, with one
sentence each on how the result here relates:
- Holtzman et al. 2019, *The Curious Case of Neural Text Degeneration* — the E6 "worm" at
  7B is a repetition loop; this is the baseline explanation to rule in or out.
- Anil et al. 2024, *Many-Shot Jailbreaking* — the closest security-relevant prior on
  in-context repetition shifting behavior, with a power law in shot count.
- Nasr et al. 2023, *Scalable Extraction of Training Data* — the repeated-token attack.
- McDougall et al. 2023, *Copy Suppression* — directly relevant to why p = 1 fails.
- Jelassi et al. 2024, *Repeat After Me* — transformers' copying capacity vs length.
- Liu et al. 2023, *Lost in the Middle* — the plan uses the term; cite it.
- Xu et al. 2022, *Learning to Break the Loop* — repetition penalties as a mitigation.
- **Done when:** each is cited at the point the paper makes the related claim, not only
  in a list.

### 0.8 Small code and dependency fixes
- Remove the dead `load_hooked` stub from `tcc/models.py:65-70` (the real one is in
  `tcc/interp.py`); fix the module docstring.
- Uncomment `transformer_lens` in `requirements.txt` (E2, E4-lite, E7, E8-lite import it).
- Pin versions in `requirements.txt` to the ones named in `results_log.md` (torch 2.8,
  transformers 4.57) and add a `requirements-lock.txt` from the next working env.
- Rename `rare_token_pool` to `short_token_pool` and fix its docstring: it takes the first
  512 short alphanumeric tokens by id, which in GPT-2 are among the most common. The
  sequence is OOD; the tokens are not. Add a TODO for a unigram-frequency pool and note the
  substring-collision risk this creates for `count_payload` at p = 1.
- Pass `attention_mask` in `e4_scale.py` and `e5_scale.py` generation calls (harmless
  today, silences the warning, correct if batching is ever added).
- Add `--greedy` to `e1_3_size_sweep.py` for parity with the other E1 scripts.
- **Done when:** `python -m tests.test_scoring` passes and the import of each experiment
  succeeds in a venv built from the pinned requirements.

### 0.9 Data durability convention, applied
`results/` is gitignored and the Track A and Track B raw JSONs exist nowhere in the repo;
only Track C was snapshotted.
- Add a `--data-dir` convention: every experiment writes its JSON to `results/` **and**
  copies it to `data/<track>/` when `--snapshot <track>` is passed; the `deploy/run_*.sh`
  scripts pass it.
- Ask the author to commit whatever Track A/B JSONs survive on the dev Mac and the Azure
  disk (`results/e1_3_trackb_t4.json`, `results/e1_4.json`, `results/e2_*.json`,
  `results/e8_capstone.json` are named in the log). Where none survive, mark the
  corresponding table rows "from run log, raw data lost."
- **Done when:** every table in `results_log.md` either links a committed JSON or carries
  that marker.

### 0.10 One source of truth for findings
Findings are restated in README, ONEPAGER, the two plan status sections, `results_log.md`,
and the paper, with drift (the paper's threats section still says "one model";
`transformer_validation_plan.md` still says E1.3 is blocked).
- `results_log.md` is the record. README and ONEPAGER each get a ≤ 10-line status that
  links to it. Plan documents lose their status sections in favor of a link.
- Fix the stale lines. Remove the status table from the paper or regenerate it from the
  log at build time.
- Tone: drop the self-descriptions of honesty and let the caveats speak. Replace
  "confirmed" / "established" with the measured quantity and its interval wherever the
  interval is wide.
- **Done when:** a finding appears in prose in at most two places (log + paper), and
  README/ONEPAGER contain only pointers and a status line per claim.

---

## Tier 1 — CPU reruns on GPT-2-small (hours, no GPU)

### 1.1 E7 with the baseline and the larger negative set (from 0.3)
Run; report induction-score AUC vs n-gram AUC vs k; false-positive rate on benign
repetition. If the n-gram baseline matches, say the detector is an n-gram repeat detector
and the induction machinery is explanatory, not operational.

### 1.2 E8-lite with the corrected controls (from 0.1)
Run; replace the table. If the random-head control now shows a comparable drop, the
"substitution is induction-carried" claim is weakened and the log must say so.

### 1.3 E1 one-shot analysis on the existing GPT-2 runs (from 0.5)
Rerun E1.1 locally (it is cheap) so the raw JSON can be committed under `data/trackA/`,
then run the floor analysis and update the E1 sections.

---

## Tier 2 — GPU reruns (RunPod L40S or equivalent; ~1 day of pod time)

### 2.1 E6 replication at power (the queued `deploy/run_trackC_p1.sh`)
Already written; not yet run. Three seeds × 20 trials for 7B serial passage; seeding-strength
sweep for the firebreak. Store per-trial `ks` (0.6). Report survival with Wilson CIs, R0
with bootstrap CIs, and the firebreak as a curve over seeding strength at **≥ 3 hops**.
Until this runs, downgrade "worm worse at scale" and "firebreak partial" in README and
paper from confirmed to preliminary.

### 2.2 E4-scale v2 (from 0.2)
≥ 30 facts, probe-based poisonability, two instruct families (Qwen2.5-7B-Instruct and
Llama-3.2-3B-Instruct), bootstrap CIs on both correlations. This is the paper's strongest
downstream claim; it should be the best-powered.

### 2.3 E5-scale with its positive control in the same run
Add a `--no-query` condition ("continue the documents") that should hijack, so the null is
demonstrably not a harness artifact. Vary the query (summarize / extract a fact / open-ended)
and add a second model. Report occupancy and an answer-corruption rate.

### 2.4 Firebreak matrix at ≥ 7B per family (P2 from the existing open list)
Only if 2.1's seeding-strength curve still shows partial transmission. Otherwise the
small-model result stands with its caveat.

---

## Tier 3 — the rewrite

Once Tier 0 and 1 are in and 2.1 has run, revise the paper around what is actually
defensible, in this order of emphasis:
1. **Methodology:** scale-matched ablation (ablate a fraction of heads, not a count) and the
   false null it corrected. This is the repo's clearest contribution.
2. **The bound:** a task query on an instruct model defuses raw-repetition hijack; the
   residual threat is semantic. State it as the main downstream result.
3. **Induction-head causality at scale**, with the one-shot copy rate as the behavioral
   measure instead of the knee.
4. **The worm**, with CIs, framed against text degeneration and many-shot priors.
5. The detector, with its trivial baseline and false-positive rate.

Abstract to ≤ 200 words. "Progress report" label stays until Tier 2 is done.

---

## What this session can and cannot do

- **Can do here:** everything in Tier 0 (code, analysis of committed JSON, docs, paper
  text), plus the per-trial storage change for 0.6.
- **Cannot do here:** any model run. This container has no torch, no GPU, and no HF cache.
  Tier 1 needs the dev Mac or any CPU box with the venv; Tier 2 needs a pod and, for
  Llama/Gemma, an `HF_TOKEN`.
- **Needed from the author:** the surviving Track A/B JSONs (0.9); a decision on whether
  to rename `rare_token_pool` (0.8, it is referenced in every experiment); and the go-ahead
  for the pod time in Tier 2.

## Suggested execution order

0.1 → 0.2 → 0.3 → 0.8 (all code, one PR) → 0.4, 0.5, 0.6 (analysis + log) → 0.7, 0.10
(writing) → Tier 1 on a CPU box → 2.1 → 2.2 → 2.3 → Tier 3.

## Progress

Each iteration is a branch off `claude/fervent-ritchie-l5m9uw`, merged back when done.

| iteration | items | state |
|---|---|---|
| 1 `…-iter1-code-fixes` | 0.1 E8 control; 0.2 E4-scale redesign; 0.3 E7 baseline + negatives; 0.8 hygiene | **merged.** Reruns (1.1, 1.2, 2.2) still needed to replace the numbers. |
| 2 `…-iter2-analysis` | 0.4 both firebreak runs + permutation test; 0.5 knee floor + one-shot inversion; 0.6 R0 from mean trajectories, per-trial storage added | **merged.** Results in `results_log.md` "Analyses of committed data". |
| 3 `…-iter3-writing` | 0.7 related work; 0.9 `--snapshot` in every experiment and the run scripts; 0.10 single source of truth (README table, ONEPAGER, plan pointers); paper text for 0.4/0.5 | **merged.** PDF not rebuilt (no TeX here). Abstract trimmed to current claims but not yet to ≤200 words (Tier 3). |
| 4 `…-iter4-prerun-code` | size sweep stores per-N curves and P(N=2) (the fix the paper already claimed); E5-scale gains three query styles, two no-query positive controls on the same contexts, per-trial storage and a control verdict (2.3's code half); `tests/test_snapshot.py` | **merged.** Pod-side only remains for 2.1–2.3. |
| 6 `…-iter6-local-reruns` | **data recovery** (all lost Track A/B raw JSON found in the dev Mac's gitignored `results/` and committed under `data/`; only Track B `e1_4` not found); **Tier 1** 1.1 (E7), 1.2 (E8-lite), 1.3 (E1.1 curves + one-shot); the **E6 per-trial rerun** on GPT-2/Pythia-160m (local complement to 2.1); PDF rebuilt (Task 7); two bug fixes in the iter4 pre-run code (`e6_worm.py` plot crash, `analysis_e6_r0.py` flat-serial format) | **done** (this branch). See the dated rerun entries in `results_log.md`. |
| Tier 1 | 1.1, 1.2, 1.3 | **done (iter6, local CPU/MPS).** 1.1: E7 detector does not beat a model-free n-gram baseline (both AUC 1.0; induction worse on benign-rep FPR). 1.2: substitution still induction-carried at m≥8 but weaker under a true random control. 1.3: E1.1 curves committed, direct P(N=2) reported with Wilson CIs. Also ran the E6 per-trial rerun (R_geo<1 everywhere; "endemic" = locked loops). **Not run:** the optional Task-6 duplicate reruns (E1.2/E1.5/E2/E5/E4-trust) — skipped because their raw data was recovered intact, so they would only add fresh sampled copies and change no claim. |
| Tier 2 | 2.1, 2.2, 2.3, 2.4 | not started: needs a GPU pod (+ `HF_TOKEN` for Llama/Gemma) |
| Tier 3 | rewrite | after Tier 2 |

Open decisions for the author: whether to keep the `rare_token_pool` alias (kept, deprecated);
pod budget for Tier 2. **Resolved (iter6):** the Track A/B raw JSON question (0.9) — all of it
survived in the dev Mac's `results/` and is now committed under `data/trackA` and `data/trackB`;
only Track B `e1_4` was not found (may still be on the Azure disk).
