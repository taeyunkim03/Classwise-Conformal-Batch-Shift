"""Does data integration actually remove covariate shift?

This directly tests an assumption made by the closest prior work (Corbetta
et al., arXiv:2410.23786, the scConform package). They assume integration has
already accounted for systematic differences between reference and query, so
that p(x|y) = q(x|y). If overlap stays near zero after integration, that
assumption does not hold in practice.

Measures reference-vs-query domain AUC and relative ESS BEFORE and AFTER
integration, with no conformal machinery involved (fast).

    pip install harmonypy
    python overlap_diagnostic.py --data data/pancreas.h5ad
    python overlap_diagnostic.py --data data/pancreas.h5ad --method combat
"""
import argparse, logging
import numpy as np
logging.getLogger("harmonypy").setLevel(logging.WARNING)
import scanpy as sc
import harmonypy as hm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from pancreas_gonogo import load_data, ess

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/pancreas.h5ad")
ap.add_argument("--n-hvg", type=int, default=2000)
ap.add_argument("--n-pcs", type=int, default=50)
ap.add_argument("--method", default="harmony", choices=["harmony", "combat", "none"])
ap.add_argument("--queries", default="inDrop3,smartseq2,celseq2,fluidigmc1")
a = ap.parse_args()

adata = load_data(a.data)
bkey = "tech" if "tech" in adata.obs else "batch"
sc.pp.highly_variable_genes(adata, n_top_genes=a.n_hvg)
ad2 = adata[:, adata.var["highly_variable"]].copy()

def pcs_raw(A):
    X = A.X
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X)
    return PCA(n_components=a.n_pcs, random_state=0).fit_transform(X)

print(f"integration method: {a.method}")
sc.pp.pca(ad2, n_comps=a.n_pcs)
if a.method == "harmony":
    # Call harmonypy directly: scanpy.external.pp.harmony_integrate transposes
    # Z_corr, which newer harmonypy versions already return as (cells x PCs),
    # producing a shape error. Orientation is checked explicitly below.
    ho = hm.run_harmony(ad2.obsm["X_pca"], ad2.obs, bkey)
    Zc = np.asarray(ho.Z_corr)
    if Zc.shape[0] != ad2.n_obs:
        Zc = Zc.T
    assert Zc.shape[0] == ad2.n_obs, f"harmony output {Zc.shape} vs {ad2.n_obs} cells"
    Z_int = Zc
elif a.method == "combat":
    sc.pp.combat(ad2, key=bkey)
    sc.pp.pca(ad2, n_comps=a.n_pcs)
    Z_int = ad2.obsm["X_pca"]
else:
    Z_int = ad2.obsm["X_pca"]
Z_raw = pcs_raw(adata[:, adata.var["highly_variable"]])

def overlap(Z, mask_q, rng):
    yd = mask_q.astype(int)
    i = rng.permutation(len(Z)); tr, te = i[:len(i)//2], i[len(i)//2:]
    d = LogisticRegression(max_iter=3000).fit(Z[tr], yd[tr])
    auc = roc_auc_score(yd[te], d.predict_proba(Z[te])[:, 1])
    ref = Z[~mask_q]
    p = np.clip(d.predict_proba(ref)[:, 1], 1e-6, 1-1e-6)
    w = p/(1-p)
    return auc, ess(w)/len(w)

ckey = "celltype" if "celltype" in adata.obs else "cell_type"
ct = adata.obs[ckey].astype(str).values

def within_class_auc(Z, mask_q, rng, min_n=50):
    """Per-class separability of reference-k vs query-k, using TRUE labels.

    This is the quantity classwise+weighting actually needs. The MARGINAL AUC
    conflates it with label shift (different class proportions alone make the
    two sets separable). If within-class AUCs are near 0.5-0.6 while the
    marginal AUC is high, the residual difference is mostly LABEL shift --
    which the Mondrian partition handles structurally, with no weighting.
    """
    out = {}
    for k in np.unique(ct):
        a_ = Z[(~mask_q) & (ct == k)]
        b_ = Z[mask_q & (ct == k)]
        if len(a_) < min_n or len(b_) < min_n:
            continue
        X = np.vstack([a_, b_]); y = np.r_[np.zeros(len(a_)), np.ones(len(b_))]
        i = rng.permutation(len(X)); tr, te = i[:len(i)//2], i[len(i)//2:]
        if len(np.unique(y[te])) < 2:
            continue
        d = LogisticRegression(max_iter=3000).fit(X[tr], y[tr])
        out[k] = roc_auc_score(y[te], d.predict_proba(X[te])[:, 1])
    return out

rng = np.random.default_rng(0)
print(f"\n{'query':<14}{'AUC raw':>10}{'AUC int':>10}{'ESS raw':>10}{'ESS int':>10}")
per_class = {}
for q in a.queries.split(","):
    m = (adata.obs[bkey].astype(str) == q).values
    if m.sum() == 0:
        continue
    ar, er = overlap(Z_raw, m, rng)
    ai, ei = overlap(Z_int, m, rng)
    print(f"{q:<14}{ar:>10.3f}{ai:>10.3f}{er:>10.4f}{ei:>10.4f}")
    per_class[q] = (within_class_auc(Z_raw, m, rng), within_class_auc(Z_int, m, rng))

print("\n" + "="*70)
print("WITHIN-CLASS AUC (ref-k vs query-k, TRUE labels, classes with >=50 both)")
print("This is what weighting must fix. ~0.5 = within-class shift is GONE.")
print("="*70)
for q, (raw, integ) in per_class.items():
    ks = sorted(set(raw) & set(integ))
    if not ks:
        continue
    print(f"\n{q}:")
    print(f"  {'cell type':<22}{'raw':>8}{'integrated':>12}")
    for k in ks:
        print(f"  {k:<22}{raw[k]:>8.3f}{integ[k]:>12.3f}")
    print(f"  {'MEAN':<22}{np.mean([raw[k] for k in ks]):>8.3f}"
          f"{np.mean([integ[k] for k in ks]):>12.3f}")
print("\nIf integrated within-class AUCs are much LOWER than the marginal AUC,")
print("the residual reference-query difference is dominated by LABEL shift,")
print("which classwise (Mondrian) conformal absorbs for free -- explaining why")
print("density-ratio weighting adds set size without adding coverage.")

print("\nIf AUC stays ~0.99 after integration, the assumption that integration")
print("removes covariate shift does NOT hold for this benchmark.")
