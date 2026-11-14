"""
Label-budget experiment: how many labeled QUERY cells per class are needed to
restore class-conditional coverage, and does source-based weighting beat it?

Motivation (from the inDrop3 pilot): domain AUC ~0.99 and relative ESS ~0.001
mean reference and query barely overlap, so importance weighting buys coverage
only by inflating prediction sets (~7 of 13 labels). Target-only classwise
calibration sidesteps the problem entirely -- it needs NO density ratio, NO
overlap, and NO positivity assumption. It is the honest competitor every
weighted method must beat.

Design
------
Query cells are split into a LABEL POOL and an EVALUATION set. Every method is
scored on the SAME evaluation cells, so the comparison is fair:

  naive         source calibration, one global threshold
  classwise     source calibration, per-class thresholds
  combined_est  source calibration + unlabeled cross-fitted within-class weights
  target_only_B target-only classwise calibration using B labeled query cells
                per class drawn from the label pool (B = 5, 10, 20, 50, ...)

Reporting: worst-class coverage is computed ONLY over classes with at least
--min-eval cells in the evaluation set. In the pilot the raw minimum was driven
by a class with ONE query cell, which is a coin flip, not a measurement.

Usage
-----
    python label_budget.py --query inDrop3 --data data/pancreas.h5ad
"""

import argparse, json
import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from conformal import (aps_score_matrix, true_label_scores, align_proba,
                       naive_conformal, classwise_conformal, combined_conformal)
from pancreas_gonogo import load_data, estimate_weights_xfit


def worst_class(included, y_eval, K, min_eval):
    n = len(y_eval)
    cov = included[np.arange(n), y_eval]
    per, counts = np.full(K, np.nan), np.bincount(y_eval, minlength=K)
    for k in range(K):
        if counts[k] > 0:
            per[k] = cov[y_eval == k].mean()
    ok = counts >= min_eval
    return {"marginal": float(cov.mean()),
            "worst_all": float(np.nanmin(per)),
            "worst_ok": float(np.nanmin(per[ok])) if ok.any() else float("nan"),
            "macro_ok": float(np.nanmean(per[ok])) if ok.any() else float("nan"),
            "set_size": float(included.sum(1).mean()),
            "per_class": per.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pancreas.h5ad")
    ap.add_argument("--query", default="inDrop3")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--min-class", type=int, default=20)
    ap.add_argument("--min-eval", type=int, default=50,
                    help="classes with fewer eval cells are excluded from "
                         "worst-class reporting (they cannot be estimated)")
    ap.add_argument("--budgets", default="5,10,20,50,100")
    ap.add_argument("--out", default="results_budget.json")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    adata = load_data(args.data)
    bkey = "tech" if "tech" in adata.obs else "batch"
    ckey = "celltype" if "celltype" in adata.obs else "cell_type"
    is_q = (adata.obs[bkey].astype(str) == args.query).values
    ref, qry = adata[~is_q].copy(), adata[is_q].copy()

    shared = sorted(set(ref.obs[ckey].astype(str)) & set(qry.obs[ckey].astype(str)))
    cnt = ref.obs[ckey].astype(str).value_counts()
    keep = sorted(c for c in shared if cnt.get(c, 0) >= args.min_class)
    ref = ref[ref.obs[ckey].astype(str).isin(keep)].copy()
    qry = qry[qry.obs[ckey].astype(str).isin(keep)].copy()
    cmap = {c: i for i, c in enumerate(keep)}
    K = len(keep)
    y_ref = ref.obs[ckey].astype(str).map(cmap).values.astype(int)
    y_qry = qry.obs[ckey].astype(str).map(cmap).values.astype(int)

    print(f"query={args.query}  ref={len(y_ref)}  query={len(y_qry)}  K={K}")
    print(f"worst-class reported over classes with >= {args.min_eval} eval cells")

    methods = ["naive", "classwise", "combined_est"] + [f"target_only_{b}" for b in budgets]
    agg = {m: [] for m in methods}
    eval_counts = np.zeros(K)

    for t in range(args.n_trials):
        rng = np.random.default_rng(500 + t)

        # ---- reference split (train / calibration) ----
        perm = rng.permutation(len(y_ref))
        n_cal = len(y_ref) // 2
        cal_i, tr_i = perm[:n_cal], perm[n_cal:]

        tr_ad = ref[tr_i].copy()
        sc.pp.highly_variable_genes(tr_ad, n_top_genes=min(args.n_hvg, tr_ad.n_vars - 1))
        genes = tr_ad.var_names[tr_ad.var["highly_variable"]]

        def dense(a):
            M = a[:, genes].X
            return np.asarray(M.toarray() if hasattr(M, "toarray") else M)

        X_tr, X_cal, X_q = dense(ref[tr_i]), dense(ref[cal_i]), dense(qry)
        pca = PCA(n_components=min(args.n_pcs, X_tr.shape[1], X_tr.shape[0] - 1),
                  random_state=t).fit(X_tr)
        Z_tr, Z_cal, Z_q = pca.transform(X_tr), pca.transform(X_cal), pca.transform(X_q)
        y_tr, y_cal = y_ref[tr_i], y_ref[cal_i]

        clf = LogisticRegression(max_iter=3000).fit(Z_tr, y_tr)
        P_cal, P_q = align_proba(clf, Z_cal, K), align_proba(clf, Z_q, K)

        # ---- query split: LABEL POOL vs EVALUATION (all methods scored on eval) ----
        qperm = rng.permutation(len(y_qry))
        half = len(qperm) // 2
        pool_i, eval_i = qperm[:half], qperm[half:]
        y_eval = y_qry[eval_i]
        eval_counts += np.bincount(y_eval, minlength=K)

        s_cal = true_label_scores(aps_score_matrix(P_cal, rng=rng), y_cal)
        s_q_all = aps_score_matrix(P_q, rng=rng)
        s_eval = s_q_all[eval_i]

        # ---- source-based methods ----
        agg["naive"].append(worst_class(
            naive_conformal(s_cal, s_eval, args.alpha), y_eval, K, args.min_eval))
        agg["classwise"].append(worst_class(
            classwise_conformal(s_cal, y_cal, s_eval, args.alpha, K), y_eval, K, args.min_eval))

        w_cf, wk_cf, w_q, wk_q, folds, _fb, _auc = estimate_weights_xfit(
            Z_cal, y_cal, Z_q[eval_i], P_q[eval_i], K, rng)
        inc = np.zeros((len(eval_i), K), bool)
        for f in range(w_cf.shape[0]):
            m = folds == f
            if m.sum():
                inc[m] = combined_conformal(s_cal, y_cal, wk_cf[f],
                                            s_eval[m], wk_q[m], args.alpha, K)
        agg["combined_est"].append(worst_class(inc, y_eval, K, args.min_eval))

        # ---- target-only classwise calibration at each budget ----
        # Labels are drawn at RANDOM from the pool within each class: the clean
        # validity benchmark. Enriched/active acquisition changes the sampling
        # law and needs separate treatment.
        for B in budgets:
            sel = []
            for k in range(K):
                idx = pool_i[y_qry[pool_i] == k]
                if len(idx) == 0:
                    continue
                sel.append(rng.choice(idx, size=min(B, len(idx)), replace=False))
            sel = np.concatenate(sel) if sel else np.array([], dtype=int)
            s_tgt = true_label_scores(s_q_all[sel], y_qry[sel])
            inc_t = classwise_conformal(s_tgt, y_qry[sel], s_eval, args.alpha, K)
            agg[f"target_only_{B}"].append(worst_class(inc_t, y_eval, K, args.min_eval))

    ec = eval_counts / args.n_trials
    print("\nmean EVAL cells per class:")
    for c, n in zip(keep, ec):
        print(f"  {c:<20}{n:>8.1f}{'   (excluded from worst-class)' if n < args.min_eval else ''}")

    print(f"\n{'method':<18}{'marginal':>10}{'worst(n>=' + str(args.min_eval) + ')':>16}"
          f"{'macro':>9}{'set size':>10}")
    summary = {}
    for m in methods:
        A = agg[m]
        summary[m] = {k: float(np.nanmean([a[k] for a in A]))
                      for k in ("marginal", "worst_all", "worst_ok", "macro_ok", "set_size")}
        s = summary[m]
        print(f"{m:<18}{s['marginal']:>10.3f}{s['worst_ok']:>16.3f}"
              f"{s['macro_ok']:>9.3f}{s['set_size']:>10.2f}")

    json.dump({"query": args.query, "classes": keep, "K": K,
               "min_eval": args.min_eval, "budgets": budgets,
               "mean_eval_counts": ec.tolist(), "summary": summary},
              open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")
    print("\nKEY COMPARISON: if target_only_B reaches nominal worst-class coverage")
    print("at a SMALL B and a MUCH smaller set size than combined_est, then under")
    print("this transfer a modest labeling budget beats unlabeled density-ratio")
    print("correction outright -- and the paper's question becomes how small B can")
    print("get, not how to estimate an unidentified ratio without overlap.")


if __name__ == "__main__":
    main()
