# Security, scope, and dual-use posture

This repository studies **context contagion** — self-reproducing and transmissible token
strings in a language model's context window — as a **defensive** research problem. The
underlying threat (context poisoning, indirect prompt injection, self-replicating prompts
against RAG and agent pipelines) already exists in the wild. The goal here is to measure
*when and why* it works, and to hand defenders a concrete quantity to watch and bound.

## What we do

- **Synthetic, non-harmful payloads only.** Payloads are out-of-distribution *nonsense*
  token strings (a unique marker, random rare-token sequences). We study the copy/lock-in
  *dynamics*. No payload carries semantic intent — no instructions, no jailbreaks, no
  exfiltration content, no malware.
- **No production targets.** Experiments run only against models we host locally or against
  rate-limited APIs under their providers' terms, on our own scaffolds. We never inject
  into, probe, or target third-party live systems, shared stores, or other users' contexts.
- **Disclosure-first.** Any result that materially sharpens a practical attack is paired
  with the corresponding detector/mitigation measurement and routed through responsible
  disclosure to affected vendors *before* public release.
- **Defense is first-class.** The toy analysis already names the defense — bound the
  longest-match trust a model places on a repeated span, independent of its usefulness —
  and every offensive measurement here is accompanied by the detector/mitigation side.

## What we do not do / will not host

- No turn-key attack tooling, no payloads with harmful semantics, no scripts that target
  or scan external systems.
- No weaponization of the worm result (e.g. self-spreading payloads aimed at real
  multi-agent deployments).

## Reporting

If you believe something in this repository enables real-world harm, or you are a vendor
affected by a finding here, please open a private report (security advisory) rather than a
public issue, and we will coordinate disclosure.

## For contributors

Read this file before extending the repo. Keep payloads OOD and meaningless; keep targets
local or rate-limited-API-with-consent; pair any new attack measurement with its defensive
counterpart. When in doubt, scope down and ask.
