"""
Pancreas cross-technology experiment.

Holds out one batch (--query) and uses all other batches as the reference.
Uses the conformal methods in conformal.py.

    pip install scanpy scikit-learn numpy scipy
    python pancreas_gonogo.py --inspect                 # ALWAYS run this first
    python pancreas_gonogo.py --query smartseq2 --split-key donor

Key design decisions
--------------------
* Probability-column alignment. sklearn returns columns only for classes
  present in the training split. `align_proba` maps them into a full
  K-column matrix in the GLOBAL label order, so absent classes cannot
  corrupt score indexing or matrix shapes.
* Split BEFORE preprocessing. HVG selection and PCA are fitted on the
  TRAINING partition only, inside the trial loop, then applied to
  calibration and query. Fitting on all reference cells would let the
  representation depend on calibration covariates, breaking the standard
  split-conformal argument (proof-breaking leakage, not necessarily
  observable undercoverage).
* Cross-fitted target weights: query folds are weighted by models fitted
  without them.
* Donor/sample-aware splitting via --split-key. Cells from one donor are
  dependent; cell-level random splitting puts related cells on both sides
  and makes finite-sample exchangeability language untenable. Without the
  key the script warns and falls back to cell-level splitting, which must
  then be reported as cell-level empirical calibration ONLY.
* Within-class ratios use TRUE source labels and SOFT target memberships.
  This is a HEURISTIC: under joint covariate+label shift q(x|Y=k) is not
  identified from labeled source + unlabeled target covariates alone.
  No exact validity is claimed for the estimated variant.
* No integration is applied by default. Integration (--integrate harmony)
  is an ABLATION arm, not the only condition.
"""

import argparse, json, os, sys
import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from conformal import (
    aps_score_matrix, true_label_scores, evaluate, align_proba,
    naive_conformal, classwise_conformal, weighted_conformal, combined_conformal,
)

DATA_URL = "https://figshare.com/ndownloader/files/24539828"
DEFAULT_PATH = "data/human_pancreas_norm_complexBatch.h5ad"


# ------------------------------------------------------------------ utilities

def _is_hdf5(path):
    """h5ad files are HDF5: signature \x89HDF\r\n\x1a\n."""
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == b"\x89HDF\r\n\x1a\n"
    except OSError:
        return False


def load_data(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # A failed download leaves a 0-byte or HTML file behind; scanpy then reuses
    # it forever and h5py reports "file signature not found", which looks like
    # data corruption but is really a failed fetch. Remove it and say so.
    if os.path.exists(path) and not _is_hdf5(path):
        size = os.path.getsize(path)
        print(f"WARNING: {path} exists but is not HDF5 ({size} bytes) -- "
              f"almost certainly a failed download. Removing and retrying.")
        os.remove(path)
    try:
        adata = sc.read(path, backup_url=DATA_URL)
    except OSError as e:
        if os.path.exists(path) and not _is_hdf5(path):
            os.remove(path)
        sys.exit(
            f"\nFailed to load {path}: {e}\n\n"
            "This is a DOWNLOAD problem, not a corrupt dataset. scanpy's fetcher\n"
            "does not always follow figshare redirects, and some HPC nodes block\n"
            "outbound traffic. Fix with:\n\n"
            "    bash fetch_data.sh\n"
            "    python pancreas_gonogo.py --inspect --data data/pancreas.h5ad\n")
    if adata.X.max() > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata


def grouped_split(groups, frac, rng, labels, K, max_tries=40):
    """Split by GROUP (donor/sample), retrying for class coverage."""
    uniq = np.unique(groups)
    best = None
    for _ in range(max_tries):
        perm = rng.permutation(uniq)
        n_cal_g = max(1, int(round(len(uniq) * frac)))
        cal_m = np.isin(groups, perm[:n_cal_g])
        tr_m = ~cal_m
        if tr_m.sum() == 0 or cal_m.sum() == 0:
            continue
        n_ok = len(set(labels[tr_m]) & set(labels[cal_m]))
        if best is None or n_ok > best[0]:
            best = (n_ok, np.where(tr_m)[0], np.where(cal_m)[0])
        if n_ok == K:
            break
    return best[1], best[2]


def ess(w):
    """Kish effective sample size of a weight vector."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0


# ------------------------------------------------------------------ estimator

def estimate_weights_xfit(Z_cal, y_cal, Z_q, P_q, K, rng, n_folds=4, min_n=30):
    """Cross-fitted ratios: TRUE source labels, SOFT target memberships.

    Returns per-fold calibration weights, per-point query weights, the fold
    assignment, and a fallback record. The rarest classes are exactly
    the ones that fall back to weight 1 (NO correction) -- report this rather
    than silently treating it as a valid estimate.
    """
    n_q = len(Z_q)
    folds = rng.permutation(n_q) % n_folds
    w_cal_f = np.ones((n_folds, len(Z_cal)))
    wk_cal_f = np.ones((n_folds, len(Z_cal), K))
    w_q = np.ones(n_q)
    wk_q = np.ones((n_q, K))
    fallback = np.zeros(K, dtype=int)
    auc_list = []

    for f in range(n_folds):
        fit_m, app_m = folds != f, folds == f
        Zq_fit, Pq_fit = Z_q[fit_m], P_q[fit_m]
        if len(Zq_fit) < 50:
            continue

        X = np.vstack([Z_cal, Zq_fit])
        yd = np.r_[np.zeros(len(Z_cal)), np.ones(len(Zq_fit))]
        dom = LogisticRegression(max_iter=3000).fit(X, yd)
        # Domain AUC ~1.0 means reference and query are perfectly separable,
        # i.e. NO OVERLAP -> importance weighting is not applicable regardless
        # of estimator quality (absolute continuity fails in practice).
        from sklearn.metrics import roc_auc_score
        held = Z_q[app_m]
        if len(held) > 10:
            Xh = np.vstack([Z_cal, held])
            yh = np.r_[np.zeros(len(Z_cal)), np.ones(len(held))]
            try:
                auc_list.append(roc_auc_score(yh, dom.predict_proba(Xh)[:, 1]))
            except ValueError:
                pass

        def r(Z, na=len(Z_cal), nb=len(Zq_fit), d=dom):
            p = np.clip(d.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
            return (p / (1 - p)) * (na / nb)

        w_cal_f[f] = r(Z_cal)
        w_q[app_m] = r(Z_q[app_m])

        for k in range(K):
            src = Z_cal[y_cal == k]                 # TRUE source labels
            soft = Pq_fit[:, k]                     # SOFT target membership
            if len(src) < min_n or soft.sum() < min_n:
                fallback[k] += 1
                continue
            X = np.vstack([src, Zq_fit])
            yd = np.r_[np.zeros(len(src)), np.ones(len(Zq_fit))]
            sw = np.r_[np.ones(len(src)), soft]
            dk = LogisticRegression(max_iter=3000).fit(X, yd, sample_weight=sw)

            def rk(Z, na=len(src), nb=float(soft.sum()), d=dk):
                p = np.clip(d.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
                return (p / (1 - p)) * (na / nb)

            wk_cal_f[f, :, k] = rk(Z_cal)
            wk_q[app_m, k] = rk(Z_q[app_m])

    return w_cal_f, wk_cal_f, w_q, wk_q, folds, fallback, \
        (float(np.mean(auc_list)) if auc_list else float("nan"))


def estimate_weights_labelaware(Z_cal, y_cal, Z_q, y_q, K, folds, min_n=30):
    """RETROSPECTIVE label-aware within-class ratio benchmark.

    Uses the benchmark's true query labels -- unavailable in deployment -- with
    cross-fitting: fold f's weights come from target labels in the OTHER folds.
    Purpose: an upper reference for what within-class ratio estimation could
    achieve with labels, sitting between the unlabeled heuristic and
    target-only classwise calibration.
    """
    n_folds = int(folds.max()) + 1
    wk_cal_f = np.ones((n_folds, len(Z_cal), K))
    wk_q = np.ones((len(Z_q), K))
    fallback = np.zeros(K, dtype=int)
    for f in range(n_folds):
        fit_m, app_m = folds != f, folds == f
        for k in range(K):
            src_k = Z_cal[y_cal == k]
            tgt_k = Z_q[fit_m][y_q[fit_m] == k]
            if len(src_k) < min_n or len(tgt_k) < min_n:
                fallback[k] += 1
                continue
            X = np.vstack([src_k, tgt_k])
            yd = np.r_[np.zeros(len(src_k)), np.ones(len(tgt_k))]
            dk = LogisticRegression(max_iter=3000).fit(X, yd)
            def rk(Z, na=len(src_k), nb=len(tgt_k), d=dk):
                p = np.clip(d.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
                return (p / (1 - p)) * (na / nb)
            wk_cal_f[f, :, k] = rk(Z_cal)
            wk_q[app_m, k] = rk(Z_q[app_m])
    return wk_cal_f, wk_q, fallback


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_PATH)
    ap.add_argument("--query", default="smartseq2")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--split-key", default=None,
                    help="obs column for donor/sample-aware splitting")
    ap.add_argument("--max-split-tries", type=int, default=40,
                    help="grouped-split retries; NOTE retries are label-adaptive "
                         "-- use 1 for a non-adaptive sensitivity run")
    ap.add_argument("--integrate", default="none",
                    choices=["none", "harmony"],
                    help="apply batch integration to the PCA space before "
                         "conformal calibration. TRANSDUCTIVE: reference+query "
                         "are integrated jointly using batch labels only (no "
                         "query CELL-TYPE labels). Legitimate at deployment, "
                         "but the representation then depends on the query, "
                         "which complicates the exchangeability argument -- "
                         "report as such.")
    ap.add_argument("--min-eval", type=int, default=50,
                    help="classes with fewer QUERY cells are excluded from "
                         "worst-class reporting -- a class with 1-2 query cells "
                         "gives a coin-flip, not a coverage estimate")
    ap.add_argument("--min-class", type=int, default=20,
                    help="drop reference classes with fewer cells than this")
    ap.add_argument("--out", default="results_pancreas.json")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    adata = load_data(args.data)

    if args.inspect:
        print("obs columns:", list(adata.obs.columns))
        for c in adata.obs.columns:
            if adata.obs[c].dtype.name in ("category", "object") and adata.obs[c].nunique() < 60:
                print(f"  {c} ({adata.obs[c].nunique()}): "
                      f"{sorted(adata.obs[c].astype(str).unique())[:25]}")
        print("X shape:", adata.shape, " X max:", float(adata.X.max()))
        print("\nPick --split-key from a donor/sample-like column above.")
        return

    bkey = "tech" if "tech" in adata.obs else "batch"
    ckey = "celltype" if "celltype" in adata.obs else "cell_type"
    is_q = (adata.obs[bkey].astype(str) == args.query).values
    if is_q.sum() == 0:
        sys.exit(f"no cells with {bkey}=={args.query}; available: "
                 f"{sorted(adata.obs[bkey].astype(str).unique())}")

    ref, qry = adata[~is_q].copy(), adata[is_q].copy()
    shared = sorted(set(ref.obs[ckey].astype(str)) & set(qry.obs[ckey].astype(str)))
    ref = ref[ref.obs[ckey].astype(str).isin(shared)].copy()
    qry = qry[qry.obs[ckey].astype(str).isin(shared)].copy()

    counts = ref.obs[ckey].astype(str).value_counts()
    keep = [c for c in shared if counts.get(c, 0) >= args.min_class]
    dropped = sorted(set(shared) - set(keep))
    if dropped:
        print(f"dropping reference classes with <{args.min_class} cells: {dropped}")
    ref = ref[ref.obs[ckey].astype(str).isin(keep)].copy()
    qry = qry[qry.obs[ckey].astype(str).isin(keep)].copy()

    classes = sorted(keep)
    cmap = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    y_ref = ref.obs[ckey].astype(str).map(cmap).values.astype(int)
    y_q = qry.obs[ckey].astype(str).map(cmap).values.astype(int)

    if args.split_key and args.split_key in ref.obs:
        groups = ref.obs[args.split_key].astype(str).values
        print(f"donor/sample-aware split on '{args.split_key}' "
              f"({len(np.unique(groups))} groups)")
    else:
        groups = None
        print("WARNING: no --split-key -> CELL-LEVEL random split. Cells from a "
              "donor are dependent; report as cell-level empirical calibration, "
              "NOT a finite-sample biological guarantee.")

    print("EVALUATION TYPE: retrospective CLOSED-SET benchmark (shared types "
          "selected using query annotations; open-set/OOD difficulty removed).")
    print(f"integration: {args.integrate}"
          + ("   (TRANSDUCTIVE: representation depends on query; report accordingly)"
             if args.integrate != "none" else ""))
    print(f"query={args.query}  ref n={len(y_ref)}  query n={len(y_q)}  K={K}")
    print("reference proportions:", np.round(np.bincount(y_ref, minlength=K)/len(y_ref), 3))
    print("query     proportions:", np.round(np.bincount(y_q, minlength=K)/len(y_q), 3))

    methods = ["naive", "classwise", "weighted", "combined_est",
               "combined_marg", "combined_lab"]
    agg = {m: {"marg": [], "pc": [], "size": []} for m in methods}
    diag = {"acc": [], "ess": [], "fallback": np.zeros(K), "full": []}

    for t in range(args.n_trials):
        rng = np.random.default_rng(100 + t)
        if groups is not None:
            tr_i, cal_i = grouped_split(groups, args.cal_frac, rng, y_ref, K,
                                        max_tries=args.max_split_tries)
        else:
            perm = rng.permutation(len(y_ref))
            n_cal = int(len(y_ref) * args.cal_frac)
            cal_i, tr_i = perm[:n_cal], perm[n_cal:]

        # --- fit representation on TRAINING cells only ---
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

        if args.integrate == "harmony":
            # Joint (transductive) integration of reference + query in PCA space.
            # Uses BATCH labels only; no query cell-type labels are touched.
            import harmonypy as hm
            import pandas as pd
            Z_all = np.vstack([Z_tr, Z_cal, Z_q])
            b_all = np.r_[ref.obs[bkey].astype(str).values[tr_i],
                          ref.obs[bkey].astype(str).values[cal_i],
                          qry.obs[bkey].astype(str).values]
            meta = pd.DataFrame({bkey: b_all})
            ho = hm.run_harmony(Z_all, meta, bkey)
            Zc = np.asarray(ho.Z_corr)
            if Zc.shape[0] != Z_all.shape[0]:   # harmonypy version orientation
                Zc = Zc.T
            n1, n2 = len(Z_tr), len(Z_tr) + len(Z_cal)
            Z_tr, Z_cal, Z_q = Zc[:n1], Zc[n1:n2], Zc[n2:]

        y_tr, y_cal = y_ref[tr_i], y_ref[cal_i]
        missing = sorted(set(range(K)) - set(np.unique(y_tr)))
        if missing and t == 0:
            print(f"  note: classes {missing} absent from a training split "
                  f"(handled by align_proba)")

        clf = LogisticRegression(max_iter=3000).fit(Z_tr, y_tr)
        P_cal, P_q = align_proba(clf, Z_cal, K), align_proba(clf, Z_q, K)
        diag["acc"].append(float((P_q.argmax(1) == y_q).mean()))

        s_cal = true_label_scores(aps_score_matrix(P_cal, rng=rng), y_cal)
        s_q = aps_score_matrix(P_q, rng=rng)

        w_cf, wk_cf, w_q, wk_q, folds, fb, dom_auc = estimate_weights_xfit(
            Z_cal, y_cal, Z_q, P_q, K, rng)
        diag.setdefault("dom_auc", []).append(dom_auc)
        wq = np.concatenate([w_cf[f] for f in range(w_cf.shape[0])])
        diag.setdefault("w_q99", []).append(float(np.percentile(wq, 99)))
        diag.setdefault("w_max", []).append(float(wq.max()))
        diag["fallback"] += fb
        diag["ess"].append(np.mean([ess(w_cf[f]) / len(y_cal)
                                    for f in range(w_cf.shape[0])]))
        diag.setdefault("class_ess", []).append(
            [np.mean([ess(wk_cf[f, y_cal == k, k]) / max(1, int((y_cal == k).sum()))
                      for f in range(wk_cf.shape[0])]) for k in range(K)])
        diag.setdefault("cal_counts", []).append(np.bincount(y_cal, minlength=K))

        out = {
            "naive": naive_conformal(s_cal, s_q, args.alpha),
            "classwise": classwise_conformal(s_cal, y_cal, s_q, args.alpha, K),
        }
        wk_lab_cf, wk_lab_q, fb_lab = estimate_weights_labelaware(
            Z_cal, y_cal, Z_q, y_q, K, folds)
        diag.setdefault("fallback_lab", np.zeros(K))
        diag["fallback_lab"] += fb_lab
        inc_w = np.zeros((len(y_q), K), bool)
        inc_c = np.zeros((len(y_q), K), bool)
        inc_cm = np.zeros((len(y_q), K), bool)
        inc_lab = np.zeros((len(y_q), K), bool)
        for f in range(w_cf.shape[0]):
            m = folds == f
            if m.sum() == 0:
                continue
            inc_w[m] = weighted_conformal(s_cal, w_cf[f], s_q[m], w_q[m], args.alpha)
            inc_c[m] = combined_conformal(s_cal, y_cal, wk_cf[f], s_q[m], wk_q[m],
                                          args.alpha, K)
            inc_cm[m] = combined_conformal(
                s_cal, y_cal, np.repeat(w_cf[f][:, None], K, 1),
                s_q[m], np.repeat(w_q[m][:, None], K, 1), args.alpha, K)
            inc_lab[m] = combined_conformal(s_cal, y_cal, wk_lab_cf[f],
                                            s_q[m], wk_lab_q[m], args.alpha, K)
        out["weighted"], out["combined_est"] = inc_w, inc_c
        out["combined_marg"], out["combined_lab"] = inc_cm, inc_lab
        diag["full"].append(float((inc_c.sum(1) == K).mean()))

        for m in methods:
            e = evaluate(out[m], y_q, K)
            agg[m]["marg"].append(e["marginal"])
            agg[m]["pc"].append(e["per_class"])
            agg[m]["size"].append(e["set_size"])

    print(f"\nbase annotator accuracy: {np.mean(diag['acc']):.3f} "
          f"(>0.98 => scenario lacks discriminative power for this question)")
    auc = np.mean(diag["dom_auc"])
    print(f"domain-classifier AUC (ref vs query): {auc:.3f}", end="")
    if auc > 0.99:
        print("   <-- NO OVERLAP: importance weighting is NOT applicable here;")
        print("       any 'coverage' from weighted methods comes from abstention"
              " (huge sets),")
        print("       not correction. Report as an overlap failure.")
    elif auc > 0.95:
        print("   <-- weak overlap; weights will be unstable")
    else:
        print("   (overlap is workable)")
    print(f"mean relative calibration ESS: {np.mean(diag['ess']):.4f}"
          f"   weight p99={np.mean(diag['w_q99']):.3g}  max={np.mean(diag['w_max']):.3g}")
    print(f"full-set (all-K) frequency:    {np.mean(diag['full']):.3f}")
    nf = 4  # folds
    fb_frac = diag["fallback"] / (args.n_trials * nf)
    print(f"per-class fallback fraction (of trialxfold instances; 1.0 = class "
          f"NEVER corrected): {np.round(fb_frac, 2).tolist()}")
    print(f"per-class WITHIN-CELL relative ESS (class-k weights on Y=k cal cells): "
          f"{np.round(np.mean(diag['class_ess'], axis=0), 3).tolist()}")
    print(f"mean calibration cells per class: "
          f"{np.round(np.mean(diag['cal_counts'], axis=0), 1).tolist()}")
    print(f"label-aware benchmark fallback fraction: "
          f"{np.round(diag.get('fallback_lab', np.zeros(K)) / (args.n_trials * nf), 2).tolist()}")

    qc = np.bincount(y_q, minlength=K)
    ok = qc >= args.min_eval
    print(f"\nclasses with >= {args.min_eval} query cells: {int(ok.sum())}/{K} "
          f"-- worst-class reported over these only")
    thin = [f"{classes[k]}({qc[k]})" for k in range(K) if not ok[k]]
    if thin:
        print(f"  excluded (too few query cells): {', '.join(thin)}")
    print(f"\ntarget = {1-args.alpha:.2f}")
    print(f"{'method':<14}{'marginal':>10}{'worst(ok)':>11}{'worst(all)':>12}"
          f"{'macro(ok)':>11}{'set size':>10}")
    summary = {}
    for m in methods:
        P = np.array(agg[m]["pc"]); pc = np.nanmean(P, 0)
        se = np.nanstd(P, 0) / np.sqrt(len(P))
        summary[m] = {"marginal": float(np.mean(agg[m]["marg"])),
                      "per_class": pc.tolist(), "per_class_split_se": se.tolist(),
                      "worst_class_all": float(np.nanmin(pc)),
                      "worst_class": float(np.nanmin(pc[ok])) if ok.any() else float("nan"),
                      "macro": float(np.nanmean(pc[ok])) if ok.any() else float("nan"),
                      "query_counts": qc.tolist(), "min_eval": args.min_eval,
                      "set_size": float(np.mean(agg[m]["size"]))}
        s = summary[m]
        print(f"{m:<14}{s['marginal']:>10.3f}{s['worst_class']:>11.3f}"
              f"{s['worst_class_all']:>12.3f}{s['macro']:>11.3f}{s['set_size']:>10.2f}")

    json.dump({"classes": classes, "query_tech": args.query,
               "split_key": args.split_key, "K": K,
               "class_counts_ref": np.bincount(y_ref, minlength=K).tolist(),
               "class_counts_query": np.bincount(y_q, minlength=K).tolist(),
               "integrate": args.integrate,
               "base_accuracy": float(np.mean(diag["acc"])),
               "rel_ess": float(np.mean(diag["ess"])),
               "domain_auc": float(np.mean(diag["dom_auc"])),
               "fallbacks": diag["fallback"].astype(int).tolist(),
               "fallbacks_lab": diag.get("fallback_lab", np.zeros(K)).astype(int).tolist(),
               "summary": summary}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")
    print("NOTE: 'combined_est' (unlabeled) and 'combined_marg' are heuristics; "
          "no exact validity claimed. 'combined_lab' uses RETROSPECTIVE query "
          "labels (cross-fitted) -- a benchmark unavailable in deployment. "
          "Success/failure of combined_marg is diagnostic evidence about the "
          "transfer, not identification of the shift mechanism.")


if __name__ == "__main__":
    main()
