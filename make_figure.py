"""Central figure.

What is plotted:
per-class coverage bars with split-variability error bars (SE across the
reference-split/randomization trials; the query law is FIXED, so these are NOT
target-population or biological uncertainty); five methods including the
marginal-ratio Mondrian; a set-size panel annotated with full-set frequency.
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(here, "results.npy"))
ap.add_argument("--figure",  default=os.path.join(here, "gonogo_figure.png"))
a = ap.parse_args()
R = np.load(a.results, allow_pickle=True).item()

K, ALPHA = 5, 0.10
methods = ["naive", "classwise", "weighted", "combined", "combined_marg"]
lab = {"naive":"naive split","classwise":"classwise (Mondrian)",
       "weighted":"weighted (marginal)","combined":"Mondrian + class-cond. wts",
       "combined_marg":"Mondrian + marginal-ratio wts"}
col = {"naive":"#B0413E","classwise":"#E4A11B","weighted":"#4C6EF5",
       "combined":"#2B8A3E","combined_marg":"#845EF7"}

PI_REF = np.array([.40,.30,.15,.10,.05])
PI_A = np.array([.824,.119,.025,.024,.008]); PI_B = np.array([.05,.10,.15,.30,.40])
panels = [("A_oracle","Regime A (covariate shift)\nORACLE ratios",PI_A),
          ("A_est","Regime A (covariate shift)\nESTIMATED ratios",PI_A),
          ("B_oracle","Regime B (joint shift)\nORACLE ratios",PI_B),
          ("B_est","Regime B (joint shift)\nESTIMATED ratios",PI_B)]

fig, axes = plt.subplots(1, 5, figsize=(25, 4.9))
xs, w = np.arange(K), 0.16
for ax,(key,title,prop) in zip(axes[:4], panels):
    res = R[key]
    for i,m in enumerate(methods):
        ax.bar(xs+(i-2)*w, res[m]["per_class"], w,
               yerr=res[m]["per_class_split_se"], capsize=1.5,
               error_kw=dict(lw=0.7), label=lab[m], color=col[m],
               edgecolor="white", linewidth=0.5)
    ax.axhline(1-ALPHA, ls="--", c="black", lw=1.3, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"c{k}\n{PI_REF[k]:.0%}→{prop[k]:.1%}" for k in range(K)],
                       fontsize=7.4)
    ax.set_ylim(0.35, 1.03); ax.set_title(title, fontsize=9.8)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
axes[0].set_ylabel("class-conditional coverage")
axes[0].legend(fontsize=6.9, loc="lower left", framealpha=0.95)
axes[0].text(0.02, 0.02, "error bars: split-variability SE (fixed query law);"
             " x: reference→target share", transform=axes[0].transAxes,
             fontsize=6.6, style="italic")

ax = axes[4]
keys=[k for k,_,_ in panels]; nm=["A orc","A est","B orc","B est"]
x4=np.arange(len(keys))
for i,m in enumerate(methods):
    vals=[R[k][m]["set_size"] for k in keys]
    ax.bar(x4+(i-2)*w, vals, w, color=col[m], edgecolor="white", linewidth=0.5)
    for xi,k in zip(x4,keys):
        ff = R[k][m]["full_frac"]
        if ff >= 0.005:
            ax.text(xi+(i-2)*w, R[k][m]["set_size"]+0.06, f"{ff:.0%}",
                    ha="center", fontsize=5.6, rotation=90)
ax.axhline(1.0, ls=":", c="gray"); ax.set_xticks(x4); ax.set_xticklabels(nm, fontsize=8.5)
ax.set_ylabel("mean set size (of 5)"); ax.set_ylim(0, 4.8)
ax.set_title("Efficiency (labels: full-set freq.)", fontsize=9.8)
ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)

fig.suptitle("Oracle within-class weighting restores simulated classwise coverage. Under covariate shift the required "
             "ratio reduces to the identifiable marginal ratio and its estimate matches the oracle;\nunder joint shift the "
             "marginal ratio is the wrong estimand (fails even as oracle) and unlabeled plug-in weighting remains below "
             "nominal at substantial set-size cost", fontsize=10.2, y=1.06)
fig.tight_layout()
fig.savefig(a.figure, dpi=175, bbox_inches="tight")
print("saved ->", a.figure)
