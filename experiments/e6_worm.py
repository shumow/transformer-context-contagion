"""
E6 -- the context-window worm (tests C6), Track-A version for the 8 GB machine.

Transmission as a corollary of reproduction. An infected host generates output in which
the payload string appears k times (its viral load). The next host READS that output as
its context (passed as TEXT -- decoded then re-tokenized, as agents/RAG actually pass it)
and continues; we count how many times the payload appears in ITS output. One hop is a map
k_in -> k_out; the epidemic is its iteration over a chain of hosts.

Two experiments, both on small models:
  A. Same-model serial passage (GPT-2), sweeping payload length p -- does the worm sustain
     across hops, and is there a critical length above which it dies?
  B. Cross-tokenizer firebreak: patient zero is GPT-2 (payload native to its tokenizer);
     then the chain continues either on GPT-2 (same tokenizer) or Pythia-160M (different
     tokenizer). Predicted: the payload TEXT, re-tokenized by a different tokenizer,
     fragments and transmits worse -- a natural firebreak.

Run:  python -m experiments.e6_worm
Writes results/e6_worm_serial.png and results/e6_worm_crosstok.png + JSON.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

from tcc import payloads


def count_payload(text, payload_text):
    if not payload_text:
        return 0
    return text.count(payload_text)


def passage_chain(hosts, S0_ids, payload_text, host0_reps, T, n_hops, rng, temperature):
    """One worm run. hosts[i] is the LM acting at hop i (hosts[0] = patient zero). The
    message between hosts is TEXT. Returns k per hop: payload occurrences in each host's
    OUTPUT (its newly generated continuation)."""
    ks = []
    lm0 = hosts[0]
    ctx_ids = list(S0_ids) * host0_reps                      # patient-zero infection
    out_ids = lm0.continue_ids(ctx_ids, T, n=1, temperature=temperature)[0]
    msg = lm0.tokenizer.decode(out_ids)
    ks.append(count_payload(msg, payload_text))
    for hop in range(1, n_hops + 1):
        lm = hosts[hop]
        ctx = lm.tokenizer.encode(msg)                       # re-tokenize the message
        out_ids = lm.continue_ids(ctx, T, n=1, temperature=temperature)[0]
        msg = lm.tokenizer.decode(out_ids)
        ks.append(count_payload(msg, payload_text))
    return ks


def experiment_A(gpt2, args, rng):
    print("\n=== A. same-model serial passage (GPT-2), sweep payload length ===")
    pool = payloads.rare_token_pool(gpt2.tokenizer)
    curves = {}
    for p in args.lengths:
        runs = []
        for _ in range(args.trials):                          # fresh payload per trial
            S = payloads.sample_payloads(pool, p, 1, rng, tokenizer=gpt2.tokenizer)[0]
            ptext = gpt2.tokenizer.decode(S)
            hosts = [gpt2] * (args.hops + 1)
            runs.append(passage_chain(hosts, S, ptext, args.host0_reps, args.T,
                                      args.hops, rng, args.temperature))
        arr = np.array(runs, float)
        survive = float(np.mean(arr[:, -1] >= 1))             # fraction alive at last hop
        curves[str(p)] = dict(mean=arr.mean(axis=0).tolist(), survive=survive)
        print(f"  p={p:2d}  mean load/hop: " +
              " -> ".join(f"{v:.1f}" for v in arr.mean(axis=0)) +
              f"   survive(last hop)={survive:.0%}")
    return curves


def experiment_B(gpt2, pythia, args, rng):
    print("\n=== B. cross-tokenizer firebreak (GPT-2 patient zero) ===")
    pool = payloads.rare_token_pool(gpt2.tokenizer)
    p = args.cross_p
    out = {}
    for label, hosts_after in [('same (GPT-2 hosts)', gpt2), ('cross (Pythia-160M hosts)', pythia)]:
        runs = []
        for _ in range(args.trials):
            S = payloads.sample_payloads(pool, p, 1, rng, tokenizer=gpt2.tokenizer)[0]
            ptext = gpt2.tokenizer.decode(S)
            hosts = [gpt2] + [hosts_after] * args.hops
            runs.append(passage_chain(hosts, S, ptext, args.host0_reps, args.T,
                                      args.hops, rng, args.temperature))
        arr = np.array(runs, float)
        out[label] = arr.mean(axis=0).tolist()
        print(f"  {label:26s} load per hop: " +
              " -> ".join(f"{v:.1f}" for v in arr.mean(axis=0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lengths', type=int, nargs='+', default=[1, 2, 3, 5, 8])
    ap.add_argument('--cross-p', type=int, default=3)
    ap.add_argument('--hops', type=int, default=6)
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--host0-reps', type=int, default=16, help='patient-zero infection strength')
    ap.add_argument('--T', type=int, default=96, help='tokens generated per host')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--prefix', default='results/e6_worm')
    args = ap.parse_args()

    try:
        from tcc.models import LM
    except ModuleNotFoundError as e:
        raise SystemExit(f"E6 needs torch/transformers: {e}")

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    gpt2 = LM('gpt2', device=args.device)
    print(f"loaded gpt2 (device={gpt2.device})")
    A = experiment_A(gpt2, args, rng)
    plot_serial(args, A)

    pythia = LM('EleutherAI/pythia-160m', device=args.device)
    print(f"loaded pythia-160m (device={pythia.device})  [{time.time()-t0:.0f}s]")
    B = experiment_B(gpt2, pythia, args, rng)
    plot_cross(args, B)

    os.makedirs(os.path.dirname(args.prefix) or '.', exist_ok=True)
    with open(f"{args.prefix}.json", 'w') as f:
        json.dump(dict(serial=A, cross=B, config=vars(args),
                       seconds=round(time.time() - t0, 1)), f, indent=2)
    print(f"\nwrote {args.prefix}.json  [{time.time()-t0:.0f}s]")


def plot_serial(args, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    lens = sorted(curves, key=int)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(lens)))
    hops = range(args.hops + 1)
    for pk, col in zip(lens, colors):
        ax.plot(hops, np.clip(curves[pk]['mean'], 0.05, None), 'o-', color=col, lw=1.9,
                label=f"p={pk} (survive {curves[pk]['survive']:.0%})")
    ax.axhline(1.0, color='k', lw=0.8, ls=':', label='extinction (load<1)')
    ax.set_yscale('log'); ax.set_xlabel('transmission hop'); ax.set_ylabel('viral load (payload copies in output)')
    ax.set_title('E6-A: same-model serial passage (GPT-2)\n'
                 'short strings sustain across hops; long strings die out')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(f"{args.prefix}_serial.png", dpi=130)
    print(f"wrote {args.prefix}_serial.png")


def plot_cross(args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    hops = range(args.hops + 1)
    for (label, curve), col, mk in zip(out.items(), ['navy', 'crimson'], ['o-', 's--']):
        ax.plot(hops, np.clip(curve, 0.05, None), mk, color=col, lw=2, label=label)
    ax.axhline(1.0, color='k', lw=0.8, ls=':', label='extinction')
    ax.set_yscale('log'); ax.set_xlabel('transmission hop'); ax.set_ylabel('viral load')
    ax.set_title(f'E6-B: cross-tokenizer firebreak (p={args.cross_p})\n'
                 'patient zero = GPT-2; re-tokenizing under Pythia fragments the payload')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(f"{args.prefix}_crosstok.png", dpi=130)
    print(f"wrote {args.prefix}_crosstok.png")


if __name__ == "__main__":
    main()
