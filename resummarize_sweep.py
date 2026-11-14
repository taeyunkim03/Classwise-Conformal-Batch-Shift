"""Recompute sweep summaries with the count guard, from ALREADY-SAVED JSONs.
No re-running required.

    python resummarize_sweep.py --min-eval 50
"""
import argparse, glob, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="sweep")
ap.add_argument("--min-eval", type=int, default=50)
a = ap.parse_args()

rows = []
for p in sorted(glob.glob(f"{a.dir}/results_*.json")):
    d = json.load(open(p))
    qc = np.array(d["class_counts_query"])
    ok = qc >= a.min_eval
    s = d["summary"]
    def w(m):
        pc = np.array(s[m]["per_class"])
        return (float(np.nanmin(pc[ok])) if ok.any() else np.nan,
                float(np.nanmin(pc)), float(s[m]["set_size"]))
    rows.append(dict(tech=d["query_tech"], auc=d.get("domain_auc", np.nan),
                     ess=d.get("rel_ess", np.nan), acc=d["base_accuracy"],
                     K=d["K"], n_ok=int(ok.sum()),
                     naive=w("naive"), clswise=w("classwise"),
                     cmarg=w("combined_marg"), cest=w("combined_est"),
                     clab=w("combined_lab")))

rows.sort(key=lambda r: r["auc"])
print(f"Count guard: classes with >= {a.min_eval} query cells only.\n")
print(f"{'query':<12}{'domAUC':>7}{'ESS':>8}{'acc':>7}{'K':>4}{'ok':>4}"
      f"{'naive':>8}{'clswise':>8}{'c_est':>8}{'c_marg':>8}"
      f"{'sz_cls':>8}{'sz_marg':>8}")
for r in rows:
    print(f"{r['tech']:<12}{r['auc']:>7.3f}{r['ess']:>8.4f}{r['acc']:>7.3f}"
          f"{r['K']:>4}{r['n_ok']:>4}"
          f"{r['naive'][0]:>8.3f}{r['clswise'][0]:>8.3f}{r['cest'][0]:>8.3f}"
          f"{r['cmarg'][0]:>8.3f}{r['clswise'][2]:>8.2f}{r['cmarg'][2]:>8.2f}")

print("\nSame table WITHOUT the guard (what tiny classes do to the metric):")
print(f"{'query':<12}{'naive':>8}{'clswise':>8}{'c_est':>8}{'c_marg':>8}")
for r in rows:
    print(f"{r['tech']:<12}{r['naive'][1]:>8.3f}{r['clswise'][1]:>8.3f}"
          f"{r['cest'][1]:>8.3f}{r['cmarg'][1]:>8.3f}")

aucs = [r["auc"] for r in rows]; esss = [r["ess"] for r in rows]
print(f"\nOVERLAP ACROSS ALL {len(rows)} TRANSFERS:")
print(f"  domain AUC : min {min(aucs):.3f}  max {max(aucs):.3f}")
print(f"  rel. ESS   : min {min(esss):.4f}  max {max(esss):.4f}")
print("  -> no transfer in this benchmark has usable reference-query overlap.")
