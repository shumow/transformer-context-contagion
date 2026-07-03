#!/usr/bin/env bash
# Pull result files off a remote pod (RunPod/Lambda/any SSH box) into data/trackC/, so the
# JSONs outlive the ephemeral pod. Recovering files needs only disk access -- a CPU-only
# restart is enough; you do NOT need the GPU back just to copy results down.
#
# Usage (host/port/key come from the pod's "Connect -> SSH over exposed TCP" panel):
#   POD=root@194.26.1.23 PORT=22042 KEY=~/.ssh/runpod deploy/fetch_results.sh
#
# Optional overrides:
#   RREPO=/workspace/transformer-context-contagion   # remote repo path (default below)
#   DEST=data/trackC                                  # local dest dir (default)
#   PATTERNS="e4_scale e5_scale e6_scale"             # result basenames to fetch (no ext)
#   EXTS="json png"                                   # extensions to try per basename
#   ALL=1                                             # ignore PATTERNS, grab every results/*.{json,png}
#
# After it runs, review data/trackC/ then: git add data/trackC && git commit && git push
set -euo pipefail

cd "$(dirname "$0")/.."                               # repo root

: "${POD:?set POD=user@host (from the pod Connect panel)}"
: "${PORT:?set PORT=<ssh port>}"
: "${KEY:?set KEY=path/to/ssh/key}"
RREPO="${RREPO:-/workspace/transformer-context-contagion}"
DEST="${DEST:-data/trackC}"
PATTERNS="${PATTERNS:-e4_scale e5_scale e6_scale}"
EXTS="${EXTS:-json png}"
KEY_EXPANDED="${KEY/#\~/$HOME}"                       # expand a leading ~

[ -f "$KEY_EXPANDED" ] || { echo "!! key not found: $KEY_EXPANDED"; exit 1; }
mkdir -p "$DEST"

SSH="ssh -p $PORT -i $KEY_EXPANDED -o StrictHostKeyChecking=accept-new"
echo "==> probing $POD:$RREPO/results ..."
if ! $SSH "$POD" "test -d '$RREPO/results'" 2>/dev/null; then
  echo "!! no results dir at $RREPO/results on the pod."
  echo "   If the repo was cloned outside the persistent volume (e.g. /root), a stop wiped it."
  echo "   Set RREPO to the real path, or re-run the experiments (Case B)."
  exit 2
fi

# Build the remote file list: everything, or the named basenames x extensions.
if [ "${ALL:-0}" = "1" ]; then
  echo "==> ALL=1: listing every results/*.{$(echo "$EXTS" | tr ' ' ',')}"
  mapfile -t FILES < <($SSH "$POD" "cd '$RREPO/results' && ls -1 $(for e in $EXTS; do printf '*.%s ' "$e"; done) 2>/dev/null" || true)
else
  FILES=()
  for base in $PATTERNS; do
    for e in $EXTS; do FILES+=("$base.$e"); done
  done
fi

[ "${#FILES[@]}" -gt 0 ] || { echo "!! nothing to fetch"; exit 3; }

got=0; missed=0
for f in "${FILES[@]}"; do
  f="${f%$'\r'}"; [ -n "$f" ] || continue
  if scp -P "$PORT" -i "$KEY_EXPANDED" -o StrictHostKeyChecking=accept-new \
        "$POD:$RREPO/results/$f" "$DEST/" 2>/dev/null; then
    echo "   fetched $f"; got=$((got+1))
  else
    echo "   (skip $f -- not on pod)"; missed=$((missed+1))
  fi
done

echo "==> done: $got fetched, $missed skipped -> $DEST/"
[ "$got" -gt 0 ] && echo "   next: git add $DEST && git commit -m 'Snapshot Track C results' && git push"
exit 0
