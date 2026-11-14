#!/usr/bin/env bash
# Fetch the scIB pancreas benchmark, trying several strategies, and VERIFY the
# result is real HDF5.
#
#   bash fetch_data.sh
#
# Background: figshare frequently refuses or stalls downloads from datacenter
# IP ranges (documented repeatedly for scikit-learn and Colab). An HPC login
# node is such an IP. A 202/403 with Length: 0 is that refusal, NOT a bad URL
# and NOT a corrupt dataset. If every strategy below fails, that is the answer,
# and the laptop->scp route at the end is the reliable fix.

set -uo pipefail
mkdir -p data
OUT="data/pancreas.h5ad"
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

is_hdf5() {
  [ -s "$1" ] && [ "$(head -c 8 "$1" | od -An -tx1 | tr -d ' \n')" = "894844460d0a1a0a" ]
}

cleanup_bad() {
  for f in data/pancreas.h5ad data/human_pancreas_norm_complexBatch.h5ad; do
    if [ -e "$f" ] && ! is_hdf5 "$f"; then rm -f "$f"; fi
  done
}

try() {  # try <label> <url>
  echo ">>> trying: $1"
  echo "    $2"
  curl -L --fail-with-body --max-time 1800 --retry 2 --retry-delay 10 \
       -A "$UA" -H "Accept: */*" -o "$OUT" "$2" 2>&1 | tail -3
  if is_hdf5 "$OUT"; then
    echo "    SUCCESS"; return 0
  fi
  echo "    failed ($(stat -c%s "$OUT" 2>/dev/null || echo 0) bytes)"
  rm -f "$OUT"; return 1
}

cleanup_bad
if is_hdf5 "$OUT"; then
  echo ">>> $OUT already valid."; ls -lh "$OUT"; exit 0
fi

# Strategy 1: canonical downloader host (NOT the figshare.com redirect form)
try "ndownloader host + browser UA" \
    "https://ndownloader.figshare.com/files/24539828" && { ls -lh "$OUT"; exit 0; }

# Strategy 2: the redirect form, with UA this time
try "figshare.com/ndownloader + browser UA" \
    "https://figshare.com/ndownloader/files/24539828" && { ls -lh "$OUT"; exit 0; }

# Strategy 3: ask the figshare API for the CURRENT download_url
echo ">>> querying figshare API for current file links"
python - <<'PY' > /tmp/fs_urls.txt 2>/dev/null
import json, urllib.request
UA = {"User-Agent": "Mozilla/5.0"}
for art in ("25953868", "12420968"):   # scIB pancreas items (newer, original)
    try:
        req = urllib.request.Request(
            f"https://api.figshare.com/v2/articles/{art}/files", headers=UA)
        files = json.load(urllib.request.urlopen(req, timeout=60))
        for f in files:
            print(f"{f.get('size',0)}\t{f.get('name','?')}\t{f.get('download_url','')}")
    except Exception:
        pass
PY

if [ -s /tmp/fs_urls.txt ]; then
  echo "    API reachable. Files found:"
  awk -F'\t' '{printf "      %-52s %10.1f MB\n", $2, $1/1048576}' /tmp/fs_urls.txt
  URL=$(grep -i -E "pancreas.*h5ad|h5ad" /tmp/fs_urls.txt | sort -t$'\t' -k1 -nr \
        | head -1 | cut -f3)
  if [ -n "${URL:-}" ]; then
    try "API-provided download_url" "$URL" && { ls -lh "$OUT"; exit 0; }
  fi
else
  echo "    API not reachable from this node either."
fi

cleanup_bad
cat <<'MSG'

>>> ALL AUTOMATED STRATEGIES FAILED.

This is almost certainly figshare refusing datacenter IP ranges, not a bad URL.
Do NOT keep retrying from the cluster. Use the laptop route -- it is ~126 MB and
takes two minutes:

  ON YOUR LAPTOP:
      curl -L -o pancreas.h5ad \
        "https://ndownloader.figshare.com/files/24539828"
      # verify it is real HDF5 (should print: pancreas.h5ad: Hierarchical Data Format)
      file pancreas.h5ad
      ls -lh pancreas.h5ad          # expect ~100-130 MB

  IF THAT ALSO FAILS, open in a browser and download manually:
      https://figshare.com/articles/dataset/scIB_pancreas_dataset/25953868

  THEN COPY UP (replace the path with your own):
      scp pancreas.h5ad you@your-cluster:/path/to/conf/data/

  THEN ON HYAK:
      python pancreas_gonogo.py --inspect --data data/pancreas.h5ad

MSG
exit 1
