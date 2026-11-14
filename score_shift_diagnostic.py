"""Is the residual shift VISIBLE TO THE SCORE?

Motivation: after integration, within-class covariate AUC stays at ~0.64-0.81,
yet classwise conformal achieves nominal coverage. Conformal validity does not
require the covariate distributions to match -- it requires the NONCONFORMITY
SCORE distributions to match within each class. Covariate shift orthogonal to
the score is harmless, and correcting it costs effective sample size for
nothing.

This script measures, per class k:
  cov_auc    separability of ref-k vs query-k in the covariate space
  score_auc  separability of ref-k vs query-k using ONLY the APS score (1-D)
  score_ks   KS statistic between the two score distributions
  d_q90      difference in the 90th-percentile score (the conformal threshold
             is a high quantile, so this is what actually moves coverage)

If cov_auc is high while score_auc ~0.5 and d_q90 ~0, the residual shift is
score-orthogonal: classwise conformal stays valid and weighting is wasted.

    python score_shift_diagnostic.py --data data/pancreas.h5ad --query smartseq2 --integrate harmony
"""
import argparse, logging
logging.getLogger("harmonypy").setLevel(logging.WARNING)
import numpy as np
import scanpy as sc
import harmonypy as hm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

from conformal import aps_score_matrix, true_label_scores, align_proba
from pancreas_gonogo import load_data

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/pancreas.h5ad")
ap.add_argument("--query", default="smartseq2")
ap.add_argument("--n-hvg", type=int, default=2000)
ap.add_argument("--n-pcs", type=int, default=50)
ap.add_argument("--integrate", default="harmony", choices=["none", "harmony"])
ap.add_argument("--min-n", type=int, default=50)
ap.add_argument("--alpha", type=float, default=0.10)
a = ap.parse_args()

adata = load_data(a.data)
bkey = "tech" if "tech" in adata.obs else "batch"
ckey = "celltype" if "celltype" in adata.obs else "cell_type"
rng = np.random.default_rng(0)

is_q = (adata.obs[bkey].astype(str) == a.query).values
ref, qry = adata[~is_q].copy(), adata[is_q].copy()
shared = sorted(set(ref.obs[ckey].astype(str)) & set(qry.obs[ckey].astype(str)))
ref = ref[ref.obs[ckey].astype(str).isin(shared)].copy()
qry = qry[qry.obs[ckey].astype(str).isin(shared)].copy()
cmap = {c: i for i, c in enumerate(shared)}; K = len(shared)
y_r = ref.obs[ckey].astype(str).map(cmap).values.astype(int)
y_q = qry.obs[ckey].astype(str).map(cmap).values.astype(int)

# train / calibration split of the reference
perm = rng.permutation(len(y_r)); n_cal = len(y_r)//2
cal_i, tr_i = perm[:n_cal], perm[n_cal:]
tr_ad = ref[tr_i].copy()
sc.pp.highly_variable_genes(tr_ad, n_top_genes=a.n_hvg)
genes = tr_ad.var_names[tr_ad.var["highly_variable"]]
def dense(A):
    M = A[:, genes].X
    return np.asarray(M.toarray() if hasattr(M, "toarray") else M)
X_tr, X_cal, X_q = dense(ref[tr_i]), dense(ref[cal_i]), dense(qry)
pca = PCA(n_components=a.n_pcs, random_state=0).fit(X_tr)
Z_tr, Z_cal, Z_q = pca.transform(X_tr), pca.transform(X_cal), pca.transform(X_q)

if a.integrate == "harmony":
    import pandas as pd
    Z_all = np.vstack([Z_tr, Z_cal, Z_q])
    b_all = np.r_[ref.obs[bkey].astype(str).values[tr_i],
                  ref.obs[bkey].astype(str).values[cal_i],
                  qry.obs[bkey].astype(str).values]
    ho = hm.run_harmony(Z_all, pd.DataFrame({bkey: b_all}), bkey)
    Zc = np.asarray(ho.Z_corr)
    if Zc.shape[0] != Z_all.shape[0]:
        Zc = Zc.T
    n1, n2 = len(Z_tr), len(Z_tr)+len(Z_cal)
    Z_tr, Z_cal, Z_q = Zc[:n1], Zc[n1:n2], Zc[n2:]

clf = LogisticRegression(max_iter=3000).fit(Z_tr, y_r[tr_i])
P_cal, P_q = align_proba(clf, Z_cal, K), align_proba(clf, Z_q, K)
s_cal = true_label_scores(aps_score_matrix(P_cal, rng=rng), y_r[cal_i])
s_q = true_label_scores(aps_score_matrix(P_q, rng=rng), y_q)
y_cal = y_r[cal_i]

def auc1d(x0, x1):
    x = np.r_[x0, x1]; y = np.r_[np.zeros(len(x0)), np.ones(len(x1))]
    return roc_auc_score(y, x)

print(f"query={a.query}  integrate={a.integrate}  "
      f"base acc={(P_q.argmax(1)==y_q).mean():.3f}")
print(f"\n{'cell type':<22}{'n_cal':>7}{'n_qry':>7}{'cov_auc':>9}"
      f"{'score_auc':>11}{'score_ks':>10}{'d_q90':>9}")
rows = []
for k in range(K):
    A_, B_ = Z_cal[y_cal == k], Z_q[y_q == k]
    if len(A_) < a.min_n or len(B_) < a.min_n:
        continue
    X = np.vstack([A_, B_]); y = np.r_[np.zeros(len(A_)), np.ones(len(B_))]
    i = rng.permutation(len(X)); tr, te = i[:len(i)//2], i[len(i)//2:]
    d = LogisticRegression(max_iter=3000).fit(X[tr], y[tr])
    cov_auc = roc_auc_score(y[te], d.predict_proba(X[te])[:, 1])
    sa, sb = s_cal[y_cal == k], s_q[y_q == k]
    sc_auc = auc1d(sa, sb)
    ks = ks_2samp(sa, sb).statistic
    dq = np.quantile(sb, 1-a.alpha) - np.quantile(sa, 1-a.alpha)
    rows.append((shared[k], len(A_), len(B_), cov_auc, sc_auc, ks, dq))
    print(f"{shared[k]:<22}{len(A_):>7}{len(B_):>7}{cov_auc:>9.3f}"
          f"{sc_auc:>11.3f}{ks:>10.3f}{dq:>9.3f}")

if rows:
    m = np.array([[r[3], r[4], r[5], r[6]] for r in rows])
    print(f"{'MEAN':<22}{'':>7}{'':>7}{m[:,0].mean():>9.3f}"
          f"{m[:,1].mean():>11.3f}{m[:,2].mean():>10.3f}{m[:,3].mean():>9.3f}")
    print(f"\ncovariate separability : {m[:,0].mean():.3f}")
    print(f"score separability     : {m[:,1].mean():.3f}  (0.5 = score sees no shift)")
    print(f"shift in the {1-a.alpha:.0%} score quantile (what sets the conformal")
    print(f"threshold)             : {m[:,3].mean():+.3f}")
    if m[:,0].mean() - m[:,1].mean() > 0.15:
        print("\n=> The residual covariate shift is LARGELY ORTHOGONAL TO THE SCORE.")
        print("   Classwise conformal stays approximately valid; density-ratio")
        print("   weighting corrects a shift the score never saw, paying effective")
        print("   sample size for nothing. Covariate-overlap diagnostics (domain")
        print("   AUC / ESS) answer 'is weighting FEASIBLE', not 'is coverage BROKEN'.")
    else:
        print("\n=> The score does see the shift; residual miscoverage should track it.")
