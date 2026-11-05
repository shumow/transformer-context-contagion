"""
Durable snapshots of experiment outputs (plan item 0.9).

`results/` is gitignored. Runs on ephemeral GPU boxes have been lost that way (every Track A
and most Track B raw files; see data/README.md). Every experiment therefore takes
`--snapshot <track>`: on exit, whatever it wrote under its `--prefix` (JSON, PNG, log) is
copied to `data/<track>/`, which IS tracked. The copy is registered with `atexit`, so it also
runs after an unhandled exception -- a checkpointed run that dies mid-way still leaves its
partial JSON in the durable location.

Usage in an experiment:
    from tcc import snapshot
    ...
    ap.add_argument('--prefix', default='results/eX')
    snapshot.add_args(ap)
    args = ap.parse_args()
    snapshot.arm(args)
"""
from __future__ import annotations
import atexit
import glob
import os
import shutil

DATA_DIR = "data"


def add_args(ap):
    ap.add_argument('--snapshot', default=None, metavar='TRACK',
                    help="also copy every file written under --prefix to data/<TRACK>/ "
                         "(git-tracked) when the run ends, e.g. --snapshot trackC")


def copy_outputs(prefix, track, verbose=True):
    """Copy results/<prefix>* into data/<track>/. Returns the list of destination paths."""
    if not track:
        return []
    dest_dir = os.path.join(DATA_DIR, track)
    os.makedirs(dest_dir, exist_ok=True)
    copied = []
    for src in sorted(glob.glob(prefix + "*")):
        if not os.path.isfile(src):
            continue
        dst = os.path.join(dest_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        copied.append(dst)
    if verbose:
        if copied:
            print(f"[snapshot] copied {len(copied)} file(s) to {dest_dir}/: "
                  + ", ".join(os.path.basename(c) for c in copied))
            print(f"[snapshot] next: git add {dest_dir} && git commit -m 'Snapshot {track} results'")
        else:
            print(f"[snapshot] nothing matched {prefix}* -- no files copied")
    return copied


def arm(args):
    """Register the copy to run at interpreter exit, if --snapshot was given."""
    track = getattr(args, "snapshot", None)
    prefix = getattr(args, "prefix", None)
    if track and prefix:
        atexit.register(copy_outputs, prefix, track)
