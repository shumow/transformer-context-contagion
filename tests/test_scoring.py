"""
Unit checks for tcc.scoring and the torch-free parts of tcc.payloads. Pure numpy -- no
model needed. Run directly (`python -m tests.test_scoring`) or under pytest.
"""
import numpy as np
from tcc import scoring, payloads


def test_perfect_reproduction():
    payload = [11, 22, 33, 44, 55]
    gen = [payload[i % 5] for i in range(40)]
    m = scoring.reproduction_metrics(gen, payload, k=1)
    assert m['transition_fidelity'] == 1.0
    assert m['run_length'] == 40
    assert m['occupancy'] == 1.0


def test_phase_shift_is_robust():
    payload = [11, 22, 33, 44, 55]
    shift = 2
    shifted = [payload[(shift + i) % 5] for i in range(30)]
    pre = [payload[(shift - 1) % 5]]
    m = scoring.reproduction_metrics(shifted, payload, k=1, prefix=pre)
    assert m['transition_fidelity'] == 1.0


def test_single_slip():
    payload = [11, 22, 33, 44, 55]
    gen = [payload[i % 5] for i in range(40)]; gen[7] = 9999
    m = scoring.reproduction_metrics(gen, payload, k=1)
    assert 0.9 < m['transition_fidelity'] < 1.0
    assert m['run_length'] == 7


def test_noise_is_low():
    payload = [11, 22, 33, 44, 55]
    rng = np.random.default_rng(1)
    noise = [int(x) for x in rng.integers(100, 200, 40)]
    assert scoring.reproduction_metrics(noise, payload, k=1)['transition_fidelity'] < 0.2


def test_order_two_payload():
    pay = [1, 2, 1, 3]                       # order-1 ambiguous, order-2 exact
    gen = [pay[i % 4] for i in range(20)]
    assert scoring.reproduction_metrics(gen, pay, k=2)['transition_fidelity'] == 1.0


def test_knee_interpolates():
    kn = scoring.knee([1, 2, 4, 8, 16], [0.0, 0.1, 0.3, 0.7, 1.0])
    assert 4 < kn < 8
    assert scoring.knee([1, 2, 4], [0.0, 0.1, 0.2]) is None


def test_payload_constructors():
    rng = np.random.default_rng(0)
    pool = list(range(100, 200))
    rb = payloads.rainbow_payload(pool, 6, rng)
    assert len(rb) == len(set(rb)) == 6
    ctx = payloads.build_context(rb, 3, prefix=[5, 6])
    assert ctx[:2] == [5, 6] and len(ctx) == 2 + 6 * 3
    assert ctx[2:8] == rb


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn(); print(f"  {fn.__name__}: OK")
    print(f"all {len(fns)} tests passed")
