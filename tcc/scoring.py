"""
Transition-level reproduction scoring, ported from the toy project's lesson: score a
generated continuation against the payload at the level of TRANSITIONS, not positions.
A single slip only phase-shifts a positional comparison and tanks all later positions
even while the model is faithfully reproducing; the honest question is whether each
generated token follows the payload's successor rule given the realized window.

Pure numpy -- no torch -- so it is unit-testable without a model. Run this file directly
for a self-test:  python -m tcc.scoring
"""
from __future__ import annotations
import math
import numpy as np


def successor_map(payload, k):
    """Cyclic order-k successor map of a payload (period p): each length-k window ->
    the token that follows it in the payload, treated cyclically."""
    p = len(payload)
    return {tuple(payload[(i + j) % p] for j in range(k)): payload[(i + k) % p]
            for i in range(p)}


def reproduction_metrics(generated, payload, k=1, prefix=None):
    """Score a generated continuation against `payload`.

    `generated` : list of token ids the model produced after the context.
    `payload`   : the cyclic payload string (list of token ids), period p = len(payload).
    `k`         : order of the successor rule (1 for a distinct-token "rainbow" payload).
    `prefix`    : the k tokens immediately before `generated` (the end of the context,
                  which is the end of the payload). Defaults to the payload's last k.

    Returns a dict:
      transition_fidelity : fraction of scored steps obeying the successor rule,
      occupancy           : fraction of generated tokens in the payload alphabet,
      run_length          : correct steps before the first slip,
      scored              : number of steps whose window was a known payload context.
    """
    p = len(payload)
    succ = successor_map(payload, k)
    alphabet = set(payload)
    if prefix is None:
        prefix = [payload[(p - k + i) % p] for i in range(k)]
    window = list(prefix[-k:])
    correct = scored = run = 0
    run_open = True
    for tok in generated:
        ctx = tuple(window[-k:])
        if ctx in succ:
            scored += 1
            ok = (tok == succ[ctx])
            correct += int(ok)
            if run_open and ok:
                run += 1
            elif run_open:
                run_open = False
        window.append(tok)
    return {
        'transition_fidelity': correct / scored if scored else 0.0,
        'occupancy': float(np.mean([t in alphabet for t in generated])) if generated else 0.0,
        'run_length': run,
        'scored': scored,
    }


def reproduces(generated, payload, k=1, prefix=None, fid_threshold=0.5):
    """Boolean: did this continuation reproduce the payload (transition fidelity above
    threshold)? The per-trial event whose mean is P(reproduce)."""
    return reproduction_metrics(generated, payload, k, prefix)['transition_fidelity'] >= fid_threshold


def knee(ns, probs, threshold=0.5):
    """Locate the condensation knee: the smallest repetition count N at which
    P(reproduce) crosses `threshold`, linearly interpolated between grid points.
    Returns a float N* (or None if the curve never crosses)."""
    ns = np.asarray(ns, float)
    probs = np.asarray(probs, float)
    order = np.argsort(ns)
    ns, probs = ns[order], probs[order]
    for i in range(1, len(ns)):
        if probs[i - 1] < threshold <= probs[i]:
            t = (threshold - probs[i - 1]) / (probs[i] - probs[i - 1])
            return float(ns[i - 1] + t * (ns[i] - ns[i - 1]))
    return float(ns[0]) if probs[0] >= threshold else None


def knee_width(ns, probs, lo=0.25, hi=0.75):
    """Knee location N* (the 0.5 crossing) and the transition WIDTH = N(hi)-N(lo). A
    small width is a sharp, condensation-like knee; a large width is a graded ramp.
    Returns (N*, width); width is None if either side crossing is absent."""
    n_star = knee(ns, probs, 0.5)
    n_lo, n_hi = knee(ns, probs, lo), knee(ns, probs, hi)
    width = (n_hi - n_lo) if (n_lo is not None and n_hi is not None) else None
    return n_star, width


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (better than normal near 0/1 and
    for small n). Returns (phat, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))) / denom
    return (phat, max(0.0, center - half), min(1.0, center + half))


# --------------------------------------------------------------------------- #
def _selftest():
    payload = [11, 22, 33, 44, 55]            # rainbow (distinct), order 1
    p = len(payload)
    # perfect cyclic reproduction starting from payload[0]
    perfect = [payload[i % p] for i in range(40)]
    m = reproduction_metrics(perfect, payload, k=1)
    assert m['transition_fidelity'] == 1.0, m
    assert m['run_length'] == 40, m
    assert m['occupancy'] == 1.0

    # starting mid-cycle (different phase) with the correct preceding token -> still
    # perfect: transition scoring does not care about absolute phase, only transitions.
    shift = 3
    shifted = [payload[(shift + i) % p] for i in range(40)]
    pre = [payload[(shift - 1) % p]]
    assert reproduction_metrics(shifted, payload, k=1, prefix=pre)['transition_fidelity'] == 1.0

    # one slip then recover: fidelity high but < 1, run_length short
    bad = list(perfect); bad[5] = 999
    mb = reproduction_metrics(bad, payload, k=1)
    assert 0.9 < mb['transition_fidelity'] < 1.0, mb
    assert mb['run_length'] == 5, mb

    # off-payload noise -> low fidelity, zero occupancy
    rng = np.random.default_rng(0)
    noise = [int(x) for x in rng.integers(100, 200, 40)]
    assert reproduction_metrics(noise, payload, k=1)['transition_fidelity'] < 0.2

    # knee: crosses 0.5 between N=4 and N=8
    assert 4 < knee([1, 2, 4, 8, 16], [0.0, 0.1, 0.3, 0.7, 1.0]) < 8

    # order-2 payload (a window of 2 disambiguates)
    pay2 = [1, 2, 1, 3]                        # order-1 ambiguous (1->2 and 1->3); order-2 exact
    perfect2 = [pay2[i % len(pay2)] for i in range(20)]
    assert reproduction_metrics(perfect2, pay2, k=2)['transition_fidelity'] == 1.0
    print("tcc.scoring self-test: OK")


if __name__ == "__main__":
    _selftest()
