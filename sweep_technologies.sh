#!/usr/bin/env bash
# Sweep query technologies to build a SHIFT-SEVERITY GRADIENT.
#
#   bash sweep_technologies.sh
#
# Rationale: the scIB pancreas benchmark contains four batches of inDrop but
# only one smartseq2, one smarter, one fluidigmc1. So the amount of overlap
# between reference and query varies enormously by which technology is held out:
#
#   inDrop1/2/3/4  -> three sibling batches remain in the reference: HIGH overlap
#   celseq2        -> related celseq remains: MODERATE overlap
#   smartseq2      -> unique full-length protocol: NO overlap (observed AUC ~1)
#   smarter        -> unique: likely low overlap
#   fluidigmc1     -> unique, small (638 cells): likely low overlap
#
# The scientific question is not "does our correction work" but "when is
# unlabeled correction possible at all, and does a cheap diagnostic predict it".
# Domain AUC and ESS are that diagnostic.

set -uo pipefail
DATA="${DATA:-data/pancreas.h5ad}"
mkdir -p sweep

for Q in inDrop3 inDrop1 celseq2 celseq smarter fluidigmc1 smartseq2; do
  echo "=============================================================="
  echo ">>> query technology: $Q"
  echo "=============================================================="
  python pancreas_gonogo.py --query "$Q" --data "$DATA" \
      --out "sweep/results_${Q}.json" 2>&1 | tee "sweep/log_${Q}.txt"
  echo
done

echo "=============================================================="
echo ">>> SUMMARY (domain AUC vs whether correction is meaningful)"
echo "=============================================================="
python - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("sweep/results_*.json")):
    d = json.load(open(p))
    s = d["summary"]
    rows.append((d["query_tech"], d.get("domain_auc", float("nan")),
                 d.get("rel_ess", float("nan")), d["base_accuracy"],
                 s["naive"]["worst_class"], s["classwise"]["worst_class"],
                 s["combined_marg"]["worst_class"], s["combined_marg"]["set_size"],
                 s["classwise"]["set_size"], d["K"]))
rows.sort(key=lambda r: r[1])
print(f"{'query':<12}{'domAUC':>8}{'ESS':>8}{'acc':>7}{'naive':>8}{'clswise':>9}"
      f"{'c_marg':>8}{'sz_cls':>8}{'sz_marg':>9}{'K':>4}")
for r in rows:
    print(f"{r[0]:<12}{r[1]:>8.3f}{r[2]:>8.4f}{r[3]:>7.3f}{r[4]:>8.3f}{r[5]:>9.3f}"
          f"{r[6]:>8.3f}{r[8]:>8.2f}{r[7]:>9.2f}{r[9]:>4d}")
print("\nRead: low domAUC = overlap exists = weighting can work.")
print("      domAUC ~1.0 = no overlap; large sz_marg means abstention, not correction.")
PY
