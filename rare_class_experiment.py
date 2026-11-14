"""
The rare-class calibration problem, and a shrinkage method for it.

WHY THIS EXPERIMENT
-------------------
Across 7 transfers, classwise conformal reaches near-nominal MACRO coverage
(0.886-0.908) -- there is no shift-correction gap left to close. But classes
with few calibration cells are a different story:

    inDrop3 calibration counts:
      alpha 2188 ... schwann 12, epsilon 16, mast 18, macrophage 31

At alpha=0.1 the classwise threshold is the ceil((n+1)(1-alpha))-th smallest
score. For n < 19 that index EXCEEDS n-1, so the threshold is the maximum score
and the class enters essentially every prediction set: pure set-size cost, zero
information. Five of thirteen pancreas cell types are in that regime -- and they
are the rare types, where annotation uncertainty matters most.

That failure has been invisible because those classes ALSO have too few query
cells to measure coverage. This script separates the two:

    subsample the CALIBRATION cells of an ABUNDANT class,
    keep ALL of its query cells.

Now calibration is scarce (the condition under study) while evaluation stays
well-powered (so coverage is measurable). Ground truth is known throughout.

THE METHOD UNDER TEST
---------------------
Shrinkage calibration: for class k, compute a weighted quantile over ALL
calibration scores, giving class-k cells weight 1 and all other cells weight
w0. w0=0 recovers classwise (unbiased per class, high variance when n_k small);
w0=1 recovers naive/marginal (low variance, biased toward the abundant classes).
Intermediate w0 trades bias against variance -- the classic shrinkage move,
which for rare classes may beat both endpoints.

    w0 fixed:      --w0 0.05
    w0 adaptive:   w0_k = c / (n_k + c)   (shrink hard when n_k is tiny,
                                           vanish as n_k grows)  --adaptive

HONESTY
-------
Shrinkage BREAKS exact classwise validity: pooling other classes' scores biases
the threshold. The claim under test is empirical -- that for small n_k it
reduces coverage ERROR (both under- and over-coverage) and set size relative to
degenerate classwise calibration. Report it as a bias-variance tradeoff, never
as a finite-sample guarantee.

    python rare_class_experiment.py --data data/pancreas.h5ad --query inDrop3 --integrate harmony
"""

import argparse, json, logging
logging.getLogger("harmonypy").setLevel(logging.WARNING)
import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from conformal import (aps_score_matrix, true_label_scores, align_proba,
                       weighted_threshold)
from pancreas_gonogo import load_data


def shrunk_thresholds(s_cal, y_cal, K, alpha, w0):
    """Per-class thresholds with weight 1 on class k and w0 on every other class."""
    taus = np.full(K, np.inf)
    for k in range(K):
        w = np.where(y_cal == k, 1.0, w0)
        if w.sum() <= 0:
            continue
        taus[k] = weighted_threshold(s_cal, w, np.ones(1), alpha)[0]
    return taus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pancreas.h5ad")
    ap.add_argument("--query", default="inDrop3")
    ap.add_argument("--integrate", default="harmony", choices=["none", "harmony"])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--sizes", default="10,15,20,30,50,100,200")
    ap.add_argument("--w0-grid", default="0.0,0.01,0.05,0.15,0.5,1.0")
    ap.add_argument("--adaptive-c", type=float, default=20.0)
    ap.add_argument("--min-query", type=int, default=150,
                    help="only classes with at least this many QUERY cells are "
                         "used as targets, so coverage stays measurable")
    ap.add_argument("--out", default="results_rare_class.json")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    w0s = [float(x) for x in args.w0_grid.split(",")]

    adata = load_data(args.data)
    bkey = "tech" if "tech" in adata.obs else "batch"
    ckey = "celltype" if "celltype" in adata.obs else "cell_type"
    is_q = (adata.obs[bkey].astype(str) == args.query).values
    ref, qry = adata[~is_q].copy(), adata[is_q].copy()
    shared = sorted(set(ref.obs[ckey].astype(str)) & set(qry.obs[ckey].astype(str)))
    cnt = ref.obs[ckey].astype(str).value_counts()
    keep = sorted(c for c in shared if cnt.get(c, 0) >= 20)
    ref = ref[ref.obs[ckey].astype(str).isin(keep)].copy()
    qry = qry[qry.obs[ckey].astype(str).isin(keep)].copy()
    cmap = {c: i for i, c in enumerate(keep)}; K = len(keep)
    y_ref = ref.obs[ckey].astype(str).map(cmap).values.astype(int)
    y_q = qry.obs[ckey].astype(str).map(cmap).values.astype(int)

    qcounts = np.bincount(y_q, minlength=K)
    targets = [k for k in range(K) if qcounts[k] >= args.min_query]
    print(f"query={args.query} integrate={args.integrate} K={K}")
    print(f"target classes (>= {args.min_query} query cells, so coverage is "
          f"measurable): {[keep[k] for k in targets]}")
    print(f"\nAt alpha={args.alpha}, classwise needs n_k >= "
          f"{int(np.ceil(1/args.alpha)) - 1 + 1} for a NON-degenerate threshold; "
          f"below that the threshold is the max score.")

    # acc[(size, w0)][k] -> list of (coverage, contributed set membership rate)
    res = {}
    for t in range(args.n_trials):
        rng = np.random.default_rng(900 + t)
        perm = rng.permutation(len(y_ref)); n_cal = len(y_ref)//2
        cal_i, tr_i = perm[:n_cal], perm[n_cal:]

        tr_ad = ref[tr_i].copy()
        sc.pp.highly_variable_genes(tr_ad, n_top_genes=min(args.n_hvg, tr_ad.n_vars-1))
        genes = tr_ad.var_names[tr_ad.var["highly_variable"]]
        def dense(A):
            M = A[:, genes].X
            return np.asarray(M.toarray() if hasattr(M, "toarray") else M)
        X_tr, X_cal, X_q = dense(ref[tr_i]), dense(ref[cal_i]), dense(qry)
        pca = PCA(n_components=min(args.n_pcs, X_tr.shape[1]), random_state=t).fit(X_tr)
        Z_tr, Z_cal, Z_q = pca.transform(X_tr), pca.transform(X_cal), pca.transform(X_q)

        if args.integrate == "harmony":
            import harmonypy as hm, pandas as pd
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

        clf = LogisticRegression(max_iter=3000).fit(Z_tr, y_ref[tr_i])
        P_cal, P_q = align_proba(clf, Z_cal, K), align_proba(clf, Z_q, K)
        s_cal_full = true_label_scores(aps_score_matrix(P_cal, rng=rng), y_ref[cal_i])
        S_q = aps_score_matrix(P_q, rng=rng)
        y_cal_full = y_ref[cal_i]

        for k in targets:
            idx_k = np.where(y_cal_full == k)[0]
            others = np.where(y_cal_full != k)[0]
            for n_k in sizes:
                if len(idx_k) < n_k:
                    continue
                sub = rng.choice(idx_k, size=n_k, replace=False)
                sel = np.concatenate([others, sub])
                s_c, y_c = s_cal_full[sel], y_cal_full[sel]
                for w0 in w0s + ["adaptive"]:
                    ww = (args.adaptive_c/(n_k+args.adaptive_c)
                          if w0 == "adaptive" else w0)
                    taus = shrunk_thresholds(s_c, y_c, K, args.alpha, ww)
                    inc_k = S_q[:, k] <= taus[k]
                    cov = inc_k[y_q == k].mean()          # coverage for class k
                    fpr = inc_k[y_q != k].mean()          # how often k is wrongly in the set
                    key = (n_k, str(w0))
                    res.setdefault(key, {}).setdefault(k, []).append((cov, fpr))

    print(f"\n{'n_cal':>7}{'w0':>10}{'coverage':>10}{'|err|':>8}{'false-incl':>12}")
    table = {}
    for (n_k, w0), per_k in sorted(res.items()):
        covs = np.array([np.mean([c for c, _ in v]) for v in per_k.values()])
        fprs = np.array([np.mean([f for _, f in v]) for v in per_k.values()])
        err = np.abs(covs - (1-args.alpha)).mean()
        table[f"{n_k}|{w0}"] = dict(coverage=float(covs.mean()),
                                    abs_err=float(err), false_incl=float(fprs.mean()))
        print(f"{n_k:>7}{w0:>10}{covs.mean():>10.3f}{err:>8.3f}{fprs.mean():>12.3f}")

    json.dump({"query": args.query, "integrate": args.integrate,
               "classes": keep, "targets": [keep[k] for k in targets],
               "table": table}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")
    print("\nREAD: 'coverage' should be ~0.90. '|err|' is mean |coverage-0.90| across")
    print("target classes -- LOWER IS BETTER, and it penalises over-coverage too.")
    print("'false-incl' is how often the class is wrongly included (set-size cost).")
    print("w0=0.0 is plain classwise. If some w0>0 gives lower |err| AND lower")
    print("false-incl at small n_cal, shrinkage beats classwise where it matters.")


if __name__ == "__main__":
    main()
