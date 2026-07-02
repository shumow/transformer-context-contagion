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

## The finding (established on GPT-2-small, confirmed to 7B across model families)

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

4. **It holds at scale, and chat models aren't immune.** The knee is low and length-robust
   from **355M to 7B across four model families** (GPT-2, Pythia, Qwen2.5, Llama-3.2) and
   **persists in instruction-tuned / chat models** — they copy the planted span rather than
   refuse. The induction-head *cause* is confirmed up to **6.9–7B in three families** (ablating
   the top ~5% of induction heads collapses reproduction; random heads don't). A **capstone**
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
- The downstream claims are established on GPT-2-small: poisonability tracks **trust, not
  usefulness**; a planted span **hijacks** generation as *entry × lock-in*; output fed back
  propagates as a **worm** with a critical string length and a **cross-tokenizer firebreak**;
  and a paired **detector** separates poisoned from clean documents at **AUC 1.0** (the
  bound-trust defense).
- **What remains:** the realistic **long-context / RAG** versions at scale (4k–128k retrieval),
  a real downstream task (not a naturalness proxy) for the trust-vs-usefulness law, and larger
  host populations for the worm.
- Reported honestly as a progress report — the *qualitative* phenomena and mechanism are solid
  across scale; absolute numbers shift with model and setup.

## Reproduce

Code and the full write-up are in this repository: the harness in `tcc/`, the experiments
in `experiments/` (`e1_condensation_knee.py`, `e1_2_cue.py`, `e2_induction.py`,
`e1_3_size_sweep.py` for the size sweep, `e8_capstone.py` for the self-extraction capstone, …),
GPU provisioning in `deploy/`, the running results in `results_log.md`, and the progress-report
paper in `latex/transformer_context_contagion.tex`.
