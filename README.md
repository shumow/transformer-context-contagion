# transformer-context-contagion

Empirical test, on real language models, of the **context-contagion** conjecture worked
out exactly on a toy Markov model in
[`trust-is-the-attack-surface`](https://github.com/shumow/trust-is-the-attack-surface).

That project showed, on a global-plus-cache Markov model where the relevant objects are
*exact*, that self-reproducing token strings (quines) exist, can be planted in a cache for
a poison budget cheapest at a condensation knee, and spread between caches as a
context-window worm with a reproduction number `R0 ≈ T/(p·k_thresh)` that falls with
string length — and that all of this is **cheapest and most transmissible under a
longest-match (induction-head) surrogate**. Every claim was stated as a *conjecture* for
real transformers. This repository is where those conjectures get confirmed, refuted, or
bounded.

This repo is **conceptually downstream** of the toy project but is not a code dependency:
the numpy Markov code does not transfer; only the analysis vocabulary does, and that lives
in the two write-ups there (`trust_is_the_attack_surface.tex`, `contagious_context.tex`).

## Status

Progress report — all six claims (C1–C6) now have empirical results; the *core* is
established across scale and the *downstream* claims are confirmed, bounded, or refined.
See [`results_log.md`](results_log.md) for the running record and
[`latex/transformer_context_contagion.tex`](latex/transformer_context_contagion.tex) for
the write-up.

- **C1 (condensation knee) / C3 (length-robustness): confirmed** end-to-end on GPT-2-small
  and **at scale, 355M–7B across four families** (GPT-2, Pythia, Qwen2.5, Llama-3.2), and
  in **instruction-tuned / chat** models (E1.1–E1.5, E1.3/E1.4 at scale).
- **C2 (induction-head mechanism): confirmed causally** — ablating the induction heads
  collapses reproduction while random-head ablation does not, from GPT-2-small to **7B in
  three families** (E2, E2-at-scale).
- **C4 (poisonability tracks trust, not usefulness): confirmed**, including with a **real
  downstream task** on an instruct model (E4-lite, E4-scale).
- **C5 (RAG hijack = entry × lock-in): established on a base-model continuation, but
  *bounded* at scale** — a real task query defuses the raw-copy hijack on an instruct model,
  leaving *semantic* injection as the residual threat (E5-lite, E5-scale).
- **C6 (context-window worm): confirmed, and *worse* at scale** — the critical string
  length *grows* with model capability (a 7B model sustains the worm where small models kill
  it); the cross-tokenizer firebreak holds only *partially* under strong seeding (E6, E6-scale).
- A paired **detector** (E7) and a **self-extraction capstone** (E8) round out the
  defensive and worst-case pictures.

See [`transformer_validation_plan.md`](transformer_validation_plan.md) (core plan, C1–C6,
E1–E6) and [`trackC_scale_plan.md`](trackC_scale_plan.md) (downstream-at-scale) for the
full phasing and threats to validity. **Caveat throughout:** sample sizes are modest
(≈4–8 payloads / facts per cell); the *qualitative* phenomena and mechanism are robust, but
absolute numbers shift with model and setup, and the Track C reversals rest on the thinnest
`n` in the project (see the open-experiments list in `trackC_scale_plan.md`).

## Safety

This is **defensive** research into a context-poisoning / self-replicating-prompt threat
that already exists against real RAG and agent pipelines. It studies the *copy dynamics*
of out-of-distribution **nonsense** payloads only — never harmful instructions or
jailbreaks — against models we host or rate-limited APIs under their terms, never against
third-party production systems. See [`SECURITY.md`](SECURITY.md) for the full posture
(synthetic non-harmful payloads, no production targets, disclosure-first, dual-use
hygiene). Read it before extending this repo.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # installs torch -- pick the wheel matching your CUDA/platform
```

GPU is recommended for the open-weights mechanistic tier (GPT-2, Pythia, Llama-3.2,
Qwen2.5, Gemma-2); the smallest models (GPT-2-small, Pythia-410M) run on CPU for
iteration.

## Run the keystone experiment

```bash
python -m experiments.e1_condensation_knee --model gpt2 --lengths 1 2 3 5 8 --reps 1 2 4 8 16 32
```

Builds an out-of-distribution token string, repeats it `N` times in context, and measures
whether the model regenerates it — the real-transformer analogue of the toy's condensation
knee. Writes `results/e1_knee.png` and a table of `P(reproduce)` vs `N` per string length.

## Layout

| Path | What |
|---|---|
| `transformer_validation_plan.md` | The core research plan: claims C1–C6, experiments E1–E6, phasing, threats to validity. |
| `trackC_scale_plan.md` | The downstream-at-scale plan (C4/C5/C6 on instruct/RAG/multi-tokenizer) and the running open-experiments list. |
| `e1_plan.md` | Detailed work plan for E1 (the condensation knee): factors, controls, statistics, staged runs. |
| `results_log.md` | Running log of every empirical run (mirrors the toy project's `contagion_notes.md`). |
| `latex/transformer_context_contagion.tex` | Progress-report write-up of the current state (built PDF alongside). |
| `tcc/payloads.py` | Token-level out-of-distribution payload construction. |
| `tcc/scoring.py` | Transition-level reproduction scoring + knee location (numpy-only, self-testing). |
| `tcc/models.py` | Minimal model loader / generation wrapper (HF; TransformerLens added for E2). |
| `tcc/interp.py` | Induction-head scoring / ablation helpers (TransformerLens) used by E2/E4/E7. |
| `experiments/e1_condensation_knee.py` | E1 keystone: P(reproduce) vs repetition count, per string length. |
| `experiments/e1_2_cue.py`, `e1_3_size_sweep.py`, `e1_5_context.py` | Cue-dependence, size sweep, and context-length/recency controls. |
| `experiments/e2_induction.py` | E2: induction-head identification + causal ablation (with `--k-frac` for scale). |
| `experiments/e4_trust.py`, `e4_scale.py` | C4: poisonability vs trust vs usefulness (proxy, then a real task at scale). |
| `experiments/e5_hijack.py`, `e5_scale.py` | C5: in-document hijack (GPT-2), then long-context instruct-RAG. |
| `experiments/e6_worm.py`, `e6_scale.py` | C6: serial-passage worm + cross-tokenizer firebreak (GPT-2, then at scale / 5-tokenizer matrix). |
| `experiments/e7_detector.py` | E7: the paired bound-trust detector (defensive complement). |
| `experiments/e8_substitution.py`, `e8_capstone.py` | E8: in-context substitution primitive, then coherent self-extraction on instruct models. |
| `deploy/` | Provider-agnostic GPU bootstrap + Azure/RunPod run scripts for the at-scale tiers. |
| `tests/test_scoring.py` | Unit checks for the scorer (no torch needed); most `experiments/*.py` also have a `--selftest`. |
| `SECURITY.md` | Defensive scope, dual-use posture, disclosure policy. |
