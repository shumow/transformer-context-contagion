# Context Contagion in Transformers — One-Pager

**The claim, in one line:** *Show an LLM a novel multi-token string twice in its context
window and it will usually keep reproducing that string on its own — and the model's
**induction heads** (its in-context copy circuit) are what cause it.*

> **Revised 2026-11-03.** An earlier version of this page described a "sharp condensation
> knee" that is "flat with scale." A re-analysis showed that for multi-token strings the knee
> sits at the lowest point the measurement grid can express (one prior occurrence), so it is a
> one-shot copy rate, not a threshold. Two at-scale results (trust-vs-usefulness with a real
> task; the detector's AUC) are withdrawn pending reruns for design reasons. Details:
> `results_log.md`, "Analyses of committed data", and `review_remediation_plan.md`.

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

## The finding (established on GPT-2-small, confirmed to 7B across model families)

1. **One occurrence is usually enough.** Inject a novel, out-of-distribution token string
   of three or more tokens and show it twice. In most models and settings tested, the model
   then regenerates it more often than not (one-shot copy rate ≈0.5–1.0). Shorter strings
   (one or two tokens) are different: they need many repetitions (≈4–10) or never lock at
   all, and that is a frequency effect, not copying. So **longer strings are easier to
   plant**, the opposite of what a naive count-based mechanism would predict. (The earlier
   "sharp knee at N≈2" wording described the grid floor, not a measured transition.)

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

4. **It holds at scale, and chat models aren't immune.** The one-shot copy rate is above
   one half in 36 of 48 cells from **355M to 7B across four model families** (GPT-2, Pythia,
   Qwen2.5, Llama-3.2), and copying **persists in instruction-tuned / chat models** — they
   need a few more occurrences but copy rather than refuse. The induction-head *cause* is
   confirmed up to **6.9–7B in three families** (ablating the top ~5% of induction heads
   collapses reproduction; random heads don't). Whether the rate *rises* with scale cannot be
   read from the stored data. A **capstone**
   probes the scariest variant — a payload that reads as **gibberish until the model decodes
   it** in context: the toy Markov model was sharply depth-limited, but a capable instruct
   model **unfolds it into coherent text and will even decode-and-execute a (benign)
   instruction**, while a surface content-filter sees only noise. The evasion is real; the
   defense is **structural** (flag the conspicuous decode-scaffold), not content inspection.

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

- Established on **GPT-2-small** end to end (behavior, mechanism, controls), then **taken to
  scale on cloud GPUs**: the knee and its length-robustness hold **355M–7B across four
  families**, **survive instruction-tuning**, and the induction-head cause is confirmed **to
  ~7B in three families**. The self-extraction **capstone** is established on instruct models.
- The downstream claims began as GPT-2-small proxies and were then pushed to their real
  settings (instruct models, long-context RAG, multi-tokenizer populations). Current standing:
  - **Trust, not usefulness.** Holds on the GPT-2-small proxy. The at-scale version with a
    real task was found to have a design flaw (its "non-repetitive" contexts repeated, and its
    poisonability measure is ~1 for any repeated span by construction); the redesigned version
    (2026-11-04, three instruct models, 32 facts) finds the same direction with a smaller
    effect: a planted span is reproduced far more often inside repetitive context than inside
    length-matched distinct prose, and usefulness never raises it — the most poisonable cell is
    repeated gibberish.
  - **RAG hijack — bounded at scale.** A recent, repeated span hijacks a *base model's
    continuation* (entry × lock-in), but on an **instruct model asked to answer a query the
    hijack vanishes (occupancy 0.00)**: the task query anchors generation away from the payload.
    Confirmed on three instruct models with positive controls (the same contexts fed as a raw
    prefix are hijacked at 0.6–1.0 occupancy); the query defuses with or without a chat
    template, though whether that is task-following or simply the payload no longer being
    the last thing in context is not yet separated. Raw-repetition hijack is a continuation
    phenomenon; the residual RAG threat is *semantic* injection, outside this project's
    benign-copy scope.
  - **Worm — re-measured at power.** With 60 chains per cell, Qwen2.5 base models (3B and
    7B) keep the payload alive in 30–50% of chains where Pythia (2.8B and 6.9B), Llama-3.2-3B
    and the instruction-tuned Qwen 7B keep it in ≤20%: a family trait, not a scale effect, and
    flat in payload length. No setting shows the payload *multiplying*: every surviving chain
    is the model stuck in a repetition loop, emitting the payload at a fixed rate. That is a
    known degeneration behavior, not an epidemic. The cross-tokenizer firebreak is withdrawn:
    no same-tokenizer advantage at one or three hops at any seeding strength (the July one-hop
    result did not replicate at n=30); which model hosts a hop matters, whether its tokenizer
    matches does not.
  - A paired **detector** flags repeated spans. Its reported perfect separation was measured
    against three clean documents with no comparison to a plain "does any n-gram repeat"
    check; it has been rewritten with a proper negative set, that baseline, and a
    false-positive rate on legitimately repetitive text, and is **awaiting a rerun**.
- **What remains:** the reruns above; the **128k-context** and **larger host-population**
  versions; and the **semantic-injection** form of the RAG threat (beyond benign OOD copy).
  See `review_remediation_plan.md`.
- A progress report: small samples (3–8 per cell, single seeds) throughout; numbers are point
  estimates unless an interval is given.

## Reproduce

Code and the full write-up are in this repository: the harness in `tcc/`, the experiments
in `experiments/` (`e1_condensation_knee.py`, `e1_2_cue.py`, `e2_induction.py`,
`e1_3_size_sweep.py` for the size sweep, `e8_capstone.py` for the self-extraction capstone, …),
GPU provisioning in `deploy/`, the running results in `results_log.md`, and the progress-report
paper in `latex/transformer_context_contagion.tex`.
