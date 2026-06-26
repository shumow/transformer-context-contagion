# Context Contagion in Transformers — One-Pager

**The claim, in one line:** *Repeat a string in an LLM's context window past a sharp
threshold and the model starts reproducing that string on its own — and the model's
**induction heads** (its in-context copy circuit) are what cause it.*

This is the empirical follow-on to a toy-model analysis
([`trust-is-the-attack-surface`](https://github.com/shumow/trust-is-the-attack-surface)),
which conjectured the behavior exactly on a Markov model; here we test it on real
transformers.

## The problem

An LLM's context window is a second, fast memory it leans on heavily (in-context learning,
RAG, agent scratchpads, tool output, chat history). Anything a model leans on, an attacker
can lean on too — *context poisoning* and *prompt injection* are live threats. A basic
question underneath them: **does repeated injected content get reproduced by the model, and
if so, by what mechanism?**

## The finding (measured on GPT-2-small, with controls)

1. **A sharp threshold — the "condensation knee."** Inject a novel, out-of-distribution
   token string and repeat it `N` times. Below a small `N` the model ignores it; above it,
   the model reliably regenerates the string. The transition is *abrupt* — a knee, not a
   gradual ramp. The threshold is small (≈2–7 repetitions) and **decreases with string
   length** (longer strings are *easier* to plant — the opposite of what a naive
   count-based mechanism would predict).

2. **It's copying, not coincidence.** Priming generation with a *non-matching* token mostly
   fails to elicit the string, while a *matching* cue reliably triggers it. Reproduction is
   **cue-dependent** — the behavioral signature of a copy mechanism, not a frequency bias.
   Controls confirm the payload is genuinely novel (zero spontaneous emission) and that the
   effect needs the *pattern*, not just the tokens.

3. **The induction heads cause it.** A standard detector flags exactly the known GPT-2
   induction heads (the attention heads that do "saw `A B` earlier, now see `A` → predict
   `B`"). **Ablating just those ~8 heads near the threshold collapses the reproduction**
   (the probability of the correct copied token drops ~80%), while ablating the same number
   of random heads barely matters. So this is the in-context copy circuit doing its normal
   job — applied to an injected payload.

## Why it matters

- It's the **mechanism behind context-poisoning persistence**: a repeated payload gets the
  model to echo it back via its own copy circuit.
- The trigger is **repetition (structure), not usefulness** — boilerplate, duplicated
  passages in a RAG retrieval dump, repeated instructions, and long verbatim quotes are
  exactly what trips it, whether or not the content helps the task.
- It is **recency-sensitive**: a payload near the end of the context is much stronger than
  one buried deep (reproduction decays as the payload is pushed back).
- The payload doesn't need to be clever or meaningful — it needs to be **repeated and
  recent** enough to cross the knee, after which the model's induction heads take over.

## Status and limits

- Established on **GPT-2-small** end to end — behavior, mechanism, and controls — and the
  knee + length-robustness **replicate in a second model family (Pythia-160M)**.
- **Not yet done:** scaling to larger / instruction-tuned models (currently capped by an
  8 GB dev machine), delivery in realistic long-context/RAG settings, and cross-host
  "worm"-style propagation (output of one model fed into another). These are the next
  stages.
- This is an early, small-scale result reported honestly as a progress report — the
  *qualitative* phenomenon and its mechanism are solid; absolute numbers will shift with
  scale.

## Reproduce

Code and the full write-up are in this repository: the harness in `tcc/`, the experiments
in `experiments/` (`e1_condensation_knee.py`, `e1_2_cue.py`, `e2_induction.py`, …), the
running results in `results_log.md`, and the progress-report paper in
`latex/transformer_context_contagion.tex`.
