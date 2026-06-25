# Plan: testing context contagion on real transformers

A research plan for taking the toy-model results of this project to real language
models. The toy work (two write-ups: `trust_is_the_attack_surface.md` and
`latex/contagious_context.tex`, plus `contagion_notes.md`) establishes, *exactly* on a
global-plus-cache Markov model, a chain of claims about self-reproducing and
transmissible context strings. Every one of them was stated as a **conjecture** for real
transformers. This document scopes the experiments that would confirm or refute that
conjecture, and — crucially — the **induction-surrogate result (milestone 3.1)** has
already de-risked it on the toy: when the count cache is replaced by the project's
longest-match predictor (PPM, an induction head built from counts), contagion becomes
cheaper to plant, length-robust, and far more transmissible. The bridge to real models
is therefore an induction-head story, and that is what these experiments target.

## 0. Scope, safety, and ethics

This is **defensive** research into a context-poisoning / self-replicating-prompt threat
that already exists against real RAG and agent pipelines (Morris-II; indirect prompt
injection). Constraints we hold ourselves to:

- **Synthetic, non-harmful payloads only.** Payloads are out-of-distribution *nonsense*
  token strings (a unique marker, random rare-token sequences). We study the *copy
  dynamics*, never a harmful instruction or jailbreak. No payload carries semantic
  intent.
- **No production targets.** All experiments run against models we host or against
  rate-limited APIs under their terms, on our own scaffolds — never against third-party
  live systems.
- **Disclosure-first.** Any finding that sharpens a practical attack is paired with the
  corresponding detection/defense measurement (the toy already hands us the defense:
  bound longest-match trust on repeated spans) and routed through responsible
  disclosure before public release.
- **Dual-use hygiene.** The repository will state the defensive purpose, withhold any
  turn-key attack tooling, and emphasize the detector/mitigation side.

## 1. The claims under test

Each maps a toy object to a transformer-measurable observable.

| # | Toy result (figure) | Real-transformer prediction | Observable |
|---|---|---|---|
| C1 | Condensation knee; per-step law `ρ+(1−ρ)pg` (`contagion_fidelity`, parent `condensation`) | A novel string repeated `N` times in context begins to be *regenerated* past a sharp threshold in `N` | P(model continues the string) vs `N` |
| C2 | Induction is the engine (`contagion_ppm`) | Reproduction is carried by **induction heads**; ablating them collapses it | activation patching / head ablation |
| C3 | Order/length brittleness under counting **vanishes** under induction (`contagion_ppm` right) | Reproduction is roughly **length-robust**: long strings need ~as few reps as short ones | reps-to-reproduce vs string length |
| C4 | Exploitability tracks **trust, not usefulness** (parent paper) | Repetitive/templated contexts are most poisonable, ~independent of task benefit | poisonability vs (attention on span) vs (task benefit) |
| C5 | Delivery = entry × lock-in; minimal-poison **bridge** (`contagion_hijack`) | In a RAG/long context, a planted span hijacks generation; cost factors into entry + lock-in | hijack rate vs poison placement/size |
| C6 | The worm: `R0 ≈ T/(p·k_thresh)`, critical length (`contagion_worm`, `contagion_ppm_worm`) | Output fed back as another context **propagates**; short strings spread, long strings die; cross-tokenizer transmission likely fails | survival across hops; R0 estimate |

C1–C3 are the core and should come first; C4 is the deepest tie to the parent paper;
C5–C6 are the applied payoff.

## 2. Models and tooling

- **Mechanistic tier (open weights, induction-head access):** GPT-2 small/medium,
  Pythia-410M/1.4B/2.8B (clean induction-head literature), Llama-3.2-1B/3B,
  Qwen2.5-0.5B/1.5B/3B, Gemma-2-2B. Start at GPT-2-small / Pythia-410M for fast
  iteration, scale up to check size-dependence.
- **Black-box tier (API):** a couple of hosted models (e.g. GPT-4o-mini-class, a Haiku-class
  model) for the white-box vs black-box contrast (C-analogue of the toy's `aTpg` gap) —
  observation/generation only.
- **Stack:** HuggingFace `transformers` + `TransformerLens` (attention, activation
  patching, head ablation), optionally `nnsight`/`pyvene` for interventions; `vLLM` for
  throughput on the `N`/length sweeps. This needs `torch` and GPU; it does **not** belong
  in this pure-numpy repo — it is a **new sibling repo/module** depending on this one only
  for the conceptual framing and the analysis vocabulary.

## 3. Experiments

### E1 — The condensation knee (keystone)
**Tests C1.** Build a context `[filler] + (S × N) + cue`, where `S` is an OOD string of
length `p` (random rare tokens), and measure whether the model's continuation reproduces
`S`. Score at the **transition level** (does each generated token follow `S`'s successor
given the realized prefix), as in the toy, not positionally. Sweep `N = 1…32` and
`p ∈ {1,2,3,5,8,16}`, at a few temperatures.
- *Predicted:* a sharp knee in `N` (low, per 3.1 — induction copies after ~1–3
  presentations), and reproduction probability rising to ~1 above it.
- *Falsified if:* no threshold (gradual or absent), i.e. real models do not lock onto
  repeated novel spans.

### E2 — Attribution to induction heads
**Tests C2.** On the open models, at an `N` just above the E1 knee, use **activation
patching** and **head ablation** to attribute the reproduction to specific attention
heads; check they are the known induction heads (QK previous-token + copy). Ablate them
and confirm reproduction collapses; ablate random heads as control.
- *Predicted:* reproduction is induction-head-mediated; ablating them removes it.
- *Falsified if:* reproduction survives induction-head ablation (some other mechanism).

### E3 — Length/order robustness
**Tests C3.** From E1, plot **reps-to-reproduce vs string length/complexity**, including
"high-order" strings (long-range dependencies, e.g. de Bruijn-like token patterns).
- *Predicted:* roughly flat in length (induction is auto-order), unlike a fixed-order
  count mechanism — the toy's `contagion_ppm` right panel.
- *Falsified if:* strongly rising (length penalty), i.e. real models behave like the
  brittle count cache.

### E4 — Trust, not usefulness
**Tests C4** (the hardest, most valuable). Construct context families that vary
*repetitiveness / template structure* (which drives induction trust) **independently of
task usefulness** (whether the repeated content actually helps a downstream task).
Measure poisonability (E1-style reproduction of a planted span) against (a) measured
attention/induction engagement on the span and (b) the span's task benefit.
- *Predicted:* poisonability tracks induction engagement, **not** benefit; boilerplate /
  retrieval-dump / long-quote contexts are maximally poisonable while contributing little.
- *Falsified if:* poisonability tracks benefit (the naive conservation law the parent
  paper already refuted on the toy).

### E5 — Delivery in a RAG/long context
**Tests C5.** Place the payload inside a *retrieved document* in a realistic
long-context/RAG prompt; the legitimate query rides its own content (the "warmed cache").
Measure hijack of generation as a function of payload placement (which "bridge" context
it attaches to) and size. Look for the entry × lock-in factorization and a
minimal-poison bridge.
- *Predicted:* a small, well-placed planted span (high-attention, on the generation
  trajectory) hijacks; poison cost factors as entry + lock-in.
- *Falsified if:* hijack requires implausibly large or specially-tuned payloads.

### E6 — Transmission (the worm) and R0
**Tests C6.** Close a loop: model output is written into a store/message that becomes
*another* context (same model = serial passage; different model = cross-model). Seed a
self-reproducing span and measure how many hops it survives, estimating an `R0` and a
**critical length**. Test **cross-tokenizer** transmission explicitly (a span optimal for
one tokenizer likely fragments under another — a predicted natural firebreak).
- *Predicted:* short spans propagate (R0 > 1), long spans die after the index host;
  same-model/same-tokenizer transmission strongest; cross-tokenizer transmission weak.
- *Falsified if:* nothing propagates beyond the index host (no worm), or propagation is
  length-independent.

## 4. Metrics and analysis
- **Reproduction fidelity** (transition-level), **knee location** `N*(p)`,
  **reps-to-reproduce**, **induction-attribution fraction** (patching effect size),
  **hijack rate**, **R0 / critical length**.
- Report with seeds, temperatures, and model sizes; treat each prediction as a
  pre-registered confirm/falsify, mirroring the toy's "stated to be attacked" discipline.

## 5. Threats to validity
- **Tokenization** dominates: "string length `p`" is in tokens, not characters; OOD spans
  may tokenize unstably. Fix payloads at the token level.
- **Instruction-tuning / RLHF** confounds: a chat model may refuse, summarize, or comment
  rather than copy. Run base (non-chat) models for the clean mechanism; treat chat models
  as a separate, more realistic condition.
- **Temperature / decoding**: lock-in is a sampling-dynamics phenomenon; sweep
  temperature and report greedy vs sampled.
- **Context-length limits** cap `N·p`; size experiments to fit, and note that longer
  contexts (more induction surface) are the dangerous regime per C4.
- **Size-dependence**: induction strength grows with scale; expect knees to move. Sweep
  model size deliberately.

## 6. Phasing and deliverables
- **Phase 0 (setup):** new sibling repo, harness on GPT-2-small/Pythia-410M, payload and
  scoring utilities, reproduce a known induction-head result as a baseline.
- **Phase 1 (core):** E1 + E2 + E3 — does the condensation knee exist, is it induction,
  is it length-robust? This is the make-or-break of the conjecture and the first
  write-up.
- **Phase 2 (depth):** E4 — trust-not-usefulness on real models, the headline scientific
  claim.
- **Phase 3 (applied):** E5 + E6 — RAG delivery and the worm/R0, with the paired detector
  (bound longest-match trust on repeated spans) and responsible disclosure.
- **Deliverables:** the harness repo, a results notebook per phase, and a paper extending
  the two toy write-ups with on-transformer measurements — confirming, refuting, or
  bounding each conjecture.

## 7. Definition of done
A clear verdict on the central conjecture: **does in-context exploitability on real
transformers track induction-style trust (the weight the copy mechanism places on a
repeated span), not task usefulness — with a measurable condensation knee, length
robustness, and a transmission R0 that falls with string length?** Either outcome is
publishable: confirmation makes the toy a predictive model of a real vulnerability;
refutation bounds where the Markov surrogate stops applying.
