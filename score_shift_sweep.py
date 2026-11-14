"""Tests whether score shift, rather than covariate shift, predicts coverage
loss.

Runs the within-class score-shift diagnostic across every technology (raw and
integrated), then correlates the score-shift magnitude against the classwise
coverage gap already measured by the pipeline. Produces the two-panel figure:

   panel A: covariate AUC   vs coverage gap   -> expected: NO relationship
   panel B: score shift     vs coverage gap   -> expected: clear relationship

If panel B holds, the practical recommendation follows directly: diagnose with a
cheap within-class SCORE comparison, not a covariate domain classifier.

    python score_shift_sweep.py --data data/pancreas.h5ad
    # expects sweep/results_<tech>.json (raw) and optionally
    #         results_<tech>_harmony.json (integrated)
"""
import argparse, glob, json, os, subprocess, sys, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/pancreas.h5ad")
ap.add_argument("--techs", default="celseq,celseq2,inDrop1,inDrop3,smartseq2,fluidigmc1")
ap.add_argument("--out", default="score_shift_sweep.json")
ap.add_argument("--figure", default="score_vs_coverage.png")
a = ap.parse_args()

def run_diag(tech, integrate):
    """Run the diagnostic and parse its MEAN row."""
    cmd = [sys.executable, "score_shift_diagnostic.py", "--data", a.data,
           "--query", tech, "--integrate", integrate]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout
    m = re.search(r"covariate separability\s*:\s*([\d.]+)", out)
    s = re.search(r"score separability\s*:\s*([\d.]+)", out)
    q = re.search(r"threshold\)\s*:\s*([+-][\d.]+)", out)
    ks = re.search(r"^MEAN.*?([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s*$",
                   out, re.M)
    if not (m and s):
        print(f"  !! parse failed for {tech}/{integrate}")
        return None
    return dict(tech=tech, integrate=integrate,
                cov_auc=float(m.group(1)), score_auc=float(s.group(1)),
                d_q90=float(q.group(1)) if q else np.nan,
                score_ks=float(ks.group(3)) if ks else np.nan)

rows = []
for tech in a.techs.split(","):
    for integ in ("none", "harmony"):
        print(f">>> {tech} / {integ}")
        d = run_diag(tech, integ)
        if d is None:
            continue
        # attach measured classwise coverage from the pipeline outputs
        path = (f"sweep/results_{tech}.json" if integ == "none"
                else f"results_{tech}_harmony.json")
        if os.path.exists(path):
            j = json.load(open(path))
            cw = j["summary"]["classwise"]
            qc = np.array(j["class_counts_query"])
            pc = np.array(cw["per_class"])
            ok = qc >= 50
            d["classwise_macro"] = float(np.nanmean(pc[ok])) if ok.any() else np.nan
            d["classwise_worst"] = float(np.nanmin(pc[ok])) if ok.any() else np.nan
            d["gap"] = 0.90 - d["classwise_macro"]
        else:
            d["classwise_macro"] = d["classwise_worst"] = d["gap"] = np.nan
            print(f"    (no pipeline result at {path}; coverage left blank)")
        rows.append(d)

json.dump(rows, open(a.out, "w"), indent=2)

print(f"\n{'tech':<12}{'integ':<9}{'cov_auc':>9}{'score_auc':>11}{'|d_q90|':>9}"
      f"{'ks':>8}{'cls_macro':>11}{'gap':>8}")
for r in rows:
    print(f"{r['tech']:<12}{r['integrate']:<9}{r['cov_auc']:>9.3f}"
          f"{r['score_auc']:>11.3f}{abs(r['d_q90']):>9.3f}{r['score_ks']:>8.3f}"
          f"{r['classwise_macro']:>11.3f}{r['gap']:>8.3f}")

good = [r for r in rows if np.isfinite(r.get("gap", np.nan))]
if len(good) >= 4:
    cov = np.array([r["cov_auc"] for r in good])
    ks = np.array([r["score_ks"] for r in good])
    dq = np.array([abs(r["d_q90"]) for r in good])
    gap = np.array([r["gap"] for r in good])
    print(f"\ncorrelation with classwise coverage gap (n={len(good)}):")
    for nm, v in (("covariate AUC", cov), ("score KS", ks), ("|d_q90|", dq)):
        print(f"  {nm:<16} r = {np.corrcoef(v, gap)[0,1]:+.3f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for axi, (v, nm) in zip(ax, ((cov, "within-class covariate AUC"),
                                 (ks, "within-class score KS"))):
        for r, x, y in zip(good, v, gap):
            c = "#B0413E" if r["integrate"] == "none" else "#2B8A3E"
            axi.scatter(x, y, c=c, s=60, zorder=3)
            axi.annotate(r["tech"], (x, y), fontsize=6.5,
                         xytext=(4, 4), textcoords="offset points")
        axi.axhline(0, ls="--", c="gray", lw=1)
        axi.set_xlabel(nm); axi.set_ylabel("classwise coverage gap (0.90 - macro)")
        axi.grid(alpha=0.25); axi.set_axisbelow(True)
        axi.set_title(f"r = {np.corrcoef(v, gap)[0,1]:+.3f}", fontsize=10)
    ax[0].scatter([], [], c="#B0413E", label="no integration")
    ax[0].scatter([], [], c="#2B8A3E", label="harmony")
    ax[0].legend(fontsize=8)
    fig.suptitle("Coverage loss tracks SCORE shift, not covariate shift", y=1.02)
    fig.tight_layout(); fig.savefig(a.figure, dpi=180, bbox_inches="tight")
    print(f"\nsaved figure -> {a.figure}")
print(f"saved -> {a.out}")
