"""
Unit checks for tcc.snapshot (pure Python). Run directly (`python -m tests.test_snapshot`)
or under pytest. Uses a temporary directory as DATA_DIR so nothing lands in data/.
"""
import argparse
import os
import tempfile
from tcc import snapshot


def _with_tmp(fn):
    old = snapshot.DATA_DIR
    with tempfile.TemporaryDirectory() as tmp:
        snapshot.DATA_DIR = os.path.join(tmp, "data")
        res_dir = os.path.join(tmp, "results")
        os.makedirs(res_dir)
        try:
            fn(tmp, res_dir)
        finally:
            snapshot.DATA_DIR = old


def test_copy_outputs_copies_prefix_matches_only():
    def body(tmp, res_dir):
        for name in ("e9_run.json", "e9_run_knee.png", "e9_run.log", "other.json"):
            open(os.path.join(res_dir, name), "w").write("x")
        copied = snapshot.copy_outputs(os.path.join(res_dir, "e9_run"), "trackZ", verbose=False)
        names = sorted(os.path.basename(c) for c in copied)
        assert names == ["e9_run.json", "e9_run.log", "e9_run_knee.png"], names
        assert all(os.path.exists(c) for c in copied)
        assert not os.path.exists(os.path.join(snapshot.DATA_DIR, "trackZ", "other.json"))
    _with_tmp(body)


def test_copy_outputs_no_track_is_noop():
    def body(tmp, res_dir):
        open(os.path.join(res_dir, "e9.json"), "w").write("x")
        assert snapshot.copy_outputs(os.path.join(res_dir, "e9"), None, verbose=False) == []
        assert not os.path.exists(snapshot.DATA_DIR)
    _with_tmp(body)


def test_copy_outputs_nothing_matching():
    def body(tmp, res_dir):
        assert snapshot.copy_outputs(os.path.join(res_dir, "nope"), "trackZ", verbose=False) == []
    _with_tmp(body)


def test_add_args_and_arm():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="results/e9")
    snapshot.add_args(ap)
    args = ap.parse_args([])
    assert args.snapshot is None
    snapshot.arm(args)                         # no track -> registers nothing, must not raise
    args = ap.parse_args(["--snapshot", "trackZ"])
    assert args.snapshot == "trackZ"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn(); print(f"  {fn.__name__}: OK")
    print(f"all {len(fns)} tests passed")
