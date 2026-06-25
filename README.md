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

Phase 0 (scaffold). The harness skeleton and the keystone experiment (E1, the condensation
knee) are stubbed; the interpretability tier (E2) is not yet wired. See
[`transformer_validation_plan.md`](transformer_validation_plan.md) for the full plan,
the claims under test (C1–C6), the experiments (E1–E6), and the phasing.

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
| `transformer_validation_plan.md` | The research plan: claims C1–C6, experiments E1–E6, phasing, threats to validity. |
| `tcc/payloads.py` | Token-level out-of-distribution payload construction. |
| `tcc/scoring.py` | Transition-level reproduction scoring + knee location (numpy-only, self-testing). |
| `tcc/models.py` | Minimal model loader / generation wrapper (HF; TransformerLens added for E2). |
| `experiments/e1_condensation_knee.py` | E1 keystone: P(reproduce) vs repetition count, per string length. |
| `tests/test_scoring.py` | Unit checks for the scorer (no torch needed). |
| `SECURITY.md` | Defensive scope, dual-use posture, disclosure policy. |
