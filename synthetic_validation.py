"""
Validate the conformal machinery on synthetic data with KNOWN density ratios.

Why this must come before real data
-----------------------------------
On real data you cannot tell a correct weighted-conformal implementation from a
subtly broken one: both produce plausible prediction sets. Here the true
densities are known, so oracle weights are exact, and the methods must behave
exactly as theory says. If they don't, the bug is in the code, not the biology.

Two shift regimes
-----------------
Regime A -- pure covariate shift, P(Y|X) invariant.
    Query is an exponential tilt of the reference in x. Label shift arises as a
    CONSEQUENCE of the covariate shift, not as a separate assumption.
    Prediction: 'weighted' restores marginal coverage but not per-class;
                'combined' restores both.

Regime B -- label shift + within-class covariate shift, P(Y|X) NOT invariant.
    Class proportions change AND each class's covariate distribution moves.
    This is the regime that matters for cross-technology single-cell transfer.
    Prediction: 'weighted' FAILS (its invariance assumption is violated);
                'classwise' FAILS (P(X|Y) moved);
                only 'combined' restores per-class coverage.
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression

from conformal import (
    aps_score_matrix, true_label_scores, evaluate,
    naive_conformal, classwise_conformal, weighted_conformal, combined_conformal,
)

# ----------------------------- configuration -----------------------------
K = 5                  # number of "cell types"
D = 10                 # feature dimension (think: PCA components)
ALPHA = 0.10           # target 90% coverage
N_TRAIN, N_CAL, N_QUERY = 4000, 2000, 2000
N_TRIALS = 100
POOL = 300_000         # pool for rejection sampling in Regime A

PI_REF = np.array([0.40, 0.30, 0.15, 0.10, 0.05])   # class 4 is rare (5%)
PI_QUERY_B = np.array([0.05, 0.10, 0.15, 0.30, 0.40])  # composition reverses

SEP = 0.8              # class separation -> ~86% accuracy, realistic for annotation
TILT = 1.10            # strength of covariate tilt in Regime A
GAMMA = 0.55           # Regime B: fraction of the way toward the confusable neighbour

rng_global = np.random.default_rng(0)
MU = rng_global.normal(size=(K, D)) * SEP
V = rng_global.normal(size=D)
V = TILT * V / np.linalg.norm(V)                       # tilt direction, Regime A
# Regime B shift: move each class a fraction GAMMA toward its NEAREST other
# class mean. This is the realistic failure mode -- a new technology makes cell
# types harder to tell apart -- and it is deliberately heterogeneous across
# classes, since each class's nearest neighbour is at a different distance.
_d = np.linalg.norm(MU[:, None, :] - MU[None, :, :], axis=2)
np.fill_diagonal(_d, np.inf)
NEIGHBOUR = _d.argmin(axis=1)
DELTA = GAMMA * (MU[NEIGHBOUR] - MU)


# ----------------------------- data generation -----------------------------

def sample_reference(n, rng):
    y = rng.choice(K, size=n, p=PI_REF)
    x = MU[y] + rng.normal(size=(n, D))
    return x, y


def sample_query_A(n, rng):
    """EXACT sampler for the exponential tilt q(x) prop-to p(x)*exp(<x,V>).

    Tilting a Gaussian mixture is available in closed form, so this is exact.
    A rejection sampler would only approximate it, because the tilted
    likelihood ratio is unbounded.

        q(x|k) = N(mu_k + V, I)
        q(k)  prop-to  p(k) * exp(mu_k^T V + ||V||^2 / 2)

    P(Y|X) is preserved exactly, so label shift arises purely as a consequence
    of the covariate tilt. Oracle ratio remains w(x) prop-to exp(<x,V>).
    """
    q = tilted_priors()
    y = rng.choice(K, size=n, p=q)
    x = MU[y] + V + rng.normal(size=(n, D))
    return x, y


def tilted_priors():
    """Exact target class priors induced by the Regime A tilt."""
    logq = np.log(PI_REF) + MU @ V + 0.5 * float(V @ V)
    q = np.exp(logq - logq.max())
    return q / q.sum()


def sample_query_B(n, rng):
    """Label shift (PI_QUERY_B) plus a per-class mean displacement DELTA."""
    y = rng.choice(K, size=n, p=PI_QUERY_B)
    x = MU[y] + DELTA[y] + rng.normal(size=(n, D))
    return x, y


# ----------------------------- oracle weights -----------------------------

def oracle_weights_A(x):
    """Marginal ratio q(x)/p(x) prop-to exp(<x,V>). Within-class ratio is the
    same function (the class-specific normalizer cancels in the weighted
    quantile), so one array serves both uses."""
    logw = x @ V
    logw -= logw.max()
    w = np.exp(logw)
    return w, np.repeat(w[:, None], K, axis=1)


def oracle_weights_B(x):
    """Exact Gaussian density ratios.

    within-class:  q(x|k)/p(x|k) = N(x; MU_k + DELTA_k, I) / N(x; MU_k, I)
    marginal:      q(x)/p(x) = sum_k PI_QUERY_B[k] N(x; MU_k+DELTA_k, I)
                               / sum_k PI_REF[k]     N(x; MU_k, I)
    """
    logp_ref = np.stack([multivariate_normal.logpdf(x, MU[k], np.eye(D)) for k in range(K)], axis=1)
    logp_qry = np.stack([multivariate_normal.logpdf(x, MU[k] + DELTA[k], np.eye(D)) for k in range(K)], axis=1)

    log_within = logp_qry - logp_ref                       # (n, K)
    within = np.exp(log_within - log_within.max(axis=0, keepdims=True))

    from scipy.special import logsumexp
    log_marg = (logsumexp(logp_qry + np.log(PI_QUERY_B), axis=1)
                - logsumexp(logp_ref + np.log(PI_REF), axis=1))
    log_marg -= log_marg.max()
    return np.exp(log_marg), within


# ----------------------------- one trial -----------------------------

def run_trial(regime, seed, use_oracle=True):
    rng = np.random.default_rng(seed)

    x_tr, y_tr = sample_reference(N_TRAIN, rng)
    x_cal, y_cal = sample_reference(N_CAL, rng)
    if regime == "control":
        x_q, y_q = sample_reference(N_QUERY, rng)      # NO shift
    elif regime == "A":
        x_q, y_q = sample_query_A(N_QUERY, rng)
    else:
        x_q, y_q = sample_query_B(N_QUERY, rng)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_tr, y_tr)

    p_cal = clf.predict_proba(x_cal)
    p_q = clf.predict_proba(x_q)

    s_cal_mat = aps_score_matrix(p_cal, randomized=True, rng=rng)
    s_q_mat = aps_score_matrix(p_q, randomized=True, rng=rng)
    s_cal = true_label_scores(s_cal_mat, y_cal)

    if use_oracle:
        # Calibration and query weights MUST come from one joint call: each
        # oracle function normalises by the max over the array it receives, and
        # the weighted threshold is NOT invariant to scaling calibration and
        # test weights by DIFFERENT constants (the test weight enters the
        # normalisation on its own). Separate calls mis-scale the test side.
        n_c = len(x_cal)
        x_all = np.vstack([x_cal, x_q])
        if regime == "control":
            w_all = np.ones(len(x_all)); wk_all = np.ones((len(x_all), K))
        elif regime == "A":
            w_all, wk_all = oracle_weights_A(x_all)
        elif regime == "B":
            w_all, wk_all = oracle_weights_B(x_all)
        w_cal, wk_cal = w_all[:n_c], wk_all[:n_c]
        w_q, wk_q = w_all[n_c:], wk_all[n_c:]
    else:
        w_cf, wk_cf, w_q, wk_q, folds = estimated_weights(
            x_cal, y_cal, x_q, clf, p_q, rng)
        # cross-fitted: each query fold uses calibration weights from the model
        # fitted WITHOUT that fold. Assemble per-fold then stitch.
        n_folds = wk_cf.shape[0]
        inc_w = np.zeros((len(y_q), K), dtype=bool)
        inc_c = np.zeros((len(y_q), K), dtype=bool)
        inc_cm = np.zeros((len(y_q), K), dtype=bool)
        for f in range(n_folds):
            m = folds == f
            if m.sum() == 0:
                continue
            inc_w[m] = weighted_conformal(s_cal, w_cf[f], s_q_mat[m], w_q[m], ALPHA)
            inc_c[m] = combined_conformal(s_cal, y_cal, wk_cf[f],
                                          s_q_mat[m], wk_q[m], ALPHA, K)
            inc_cm[m] = combined_conformal(
                s_cal, y_cal, np.repeat(w_cf[f][:, None], K, 1),
                s_q_mat[m], np.repeat(w_q[m][:, None], K, 1), ALPHA, K)
        res = {}
        res["naive"] = evaluate(naive_conformal(s_cal, s_q_mat, ALPHA), y_q, K)
        res["classwise"] = evaluate(
            classwise_conformal(s_cal, y_cal, s_q_mat, ALPHA, K), y_q, K)
        res["weighted"] = evaluate(inc_w, y_q, K)
        res["combined"] = evaluate(inc_c, y_q, K)
        res["combined_marg"] = evaluate(inc_cm, y_q, K)
        return res

    res = {}
    res["naive"] = evaluate(naive_conformal(s_cal, s_q_mat, ALPHA), y_q, K)
    res["classwise"] = evaluate(classwise_conformal(s_cal, y_cal, s_q_mat, ALPHA, K), y_q, K)
    res["weighted"] = evaluate(weighted_conformal(s_cal, w_cal, s_q_mat, w_q, ALPHA), y_q, K)
    res["combined"] = evaluate(
        combined_conformal(s_cal, y_cal, wk_cal, s_q_mat, wk_q, ALPHA, K), y_q, K)
    # Marginal-ratio Mondrian: repeat the MARGINAL ratio
    # across classes. Under pure covariate shift, q(x|k)/p(x|k) equals the
    # marginal ratio up to a class constant that cancels in the Mondrian cell,
    # so this is the correct estimand in Regime A and the WRONG one in Regime B.
    res["combined_marg"] = evaluate(
        combined_conformal(s_cal, y_cal, np.repeat(w_cal[:, None], K, 1),
                           s_q_mat, np.repeat(w_q[:, None], K, 1), ALPHA, K), y_q, K)
    return res


def estimated_weights(x_cal, y_cal, x_q, clf, p_q, rng, n_folds=4):
    """Cross-fitted within-class density-ratio estimation.

    Three design choices:

    1. SOURCE bins use TRUE labels. Binning calibration cells by
       clf.predict(x_cal) would estimate q(x|Yhat=k)/p(x|Yhat=k) and then
       apply it to cells selected by their TRUE label -- two different
       conditioning events with no validity argument. True source labels are
       available, so they are used.

    2. TARGET side uses SOFT memberships p(k|x) as sample weights rather than a
       hard argmax bin. Presented as a heuristic: soft memberships are NOT a
       consistent estimator of the true-class conditional without assumptions
       about target posterior calibration / mixture identifiability.

    3. CROSS-FITTING: query weights for a fold are estimated from the other
       folds, removing the transductive dependence where query points trained
       the estimator that evaluates them.

    NONE of this resolves identifiability. Under joint covariate + label shift,
    q(x|Y=k) is not identified from labeled source + unlabeled target covariates
    alone. This is a heuristic plug-in, not a corrected estimator.
    """
    from sklearn.linear_model import LogisticRegression as LR

    n_q = len(x_q)
    folds = rng.permutation(n_q) % n_folds
    wk_cal_folds = np.ones((n_folds, len(x_cal), K))
    wk_q = np.ones((n_q, K))
    w_cal_folds = np.ones((n_folds, len(x_cal)))
    w_q = np.ones(n_q)

    for f in range(n_folds):
        fit_mask = folds != f          # estimate on OTHER folds
        app_mask = folds == f          # apply to held-out fold
        xq_fit = x_q[fit_mask]
        if len(xq_fit) < 50:
            continue

        # marginal ratio q(x)/p(x)
        X = np.vstack([x_cal, xq_fit])
        yd = np.r_[np.zeros(len(x_cal)), np.ones(len(xq_fit))]
        dom = LR(max_iter=2000).fit(X, yd)
        def r_marg(Z, na=len(x_cal), nb=len(xq_fit), d=dom):
            pr = np.clip(d.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
            return (pr / (1 - pr)) * (na / nb)
        w_cal_folds[f] = r_marg(x_cal)
        w_q[app_mask] = r_marg(x_q[app_mask])

        # within-class ratios: TRUE source labels vs SOFT target memberships
        pq_fit = p_q[fit_mask]
        for k in range(K):
            src = x_cal[y_cal == k]                    # TRUE labels
            soft = pq_fit[:, k]                        # soft membership
            if len(src) < 30 or soft.sum() < 30:
                continue                               # fallback: weight 1
            X = np.vstack([src, xq_fit])
            yd = np.r_[np.zeros(len(src)), np.ones(len(xq_fit))]
            sw = np.r_[np.ones(len(src)), soft]
            dk = LR(max_iter=2000).fit(X, yd, sample_weight=sw)
            eff_b = soft.sum()
            def rk(Z, na=len(src), nb=eff_b, d=dk):
                pr = np.clip(d.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
                return (pr / (1 - pr)) * (na / nb)
            wk_cal_folds[f, :, k] = rk(x_cal)
            wk_q[app_mask, k] = rk(x_q[app_mask])

    return w_cal_folds, wk_cal_folds, w_q, wk_q, folds


# ----------------------------- driver -----------------------------

def aggregate(regime, use_oracle):
    methods = ["naive", "classwise", "weighted", "combined", "combined_marg"]
    acc = {m: {"marg": [], "pc": [], "size": [], "full": []} for m in methods}
    for t in range(N_TRIALS):
        r = run_trial(regime, seed=1000 + t, use_oracle=use_oracle)
        for m in methods:
            acc[m]["marg"].append(r[m]["marginal"])
            acc[m]["pc"].append(r[m]["per_class"])
            acc[m]["size"].append(r[m]["set_size"])
            acc[m]["full"].append(r[m]["full_frac"])
    out = {}
    for m in methods:
        P = np.array(acc[m]["pc"])
        out[m] = {
            "marginal": float(np.mean(acc[m]["marg"])),
            "per_class": np.nanmean(P, axis=0),
            # split-variability spread, NOT target-population uncertainty
            # (fixed query law; trials vary splits/randomization only)
            "per_class_split_se": (np.nanstd(P, axis=0) / np.sqrt(len(P))),
            "pc_trials": P,
            "worst_class": float(np.nanmin(np.nanmean(P, axis=0))),
            "set_size": float(np.mean(acc[m]["size"])),
            "full_frac": float(np.mean(acc[m]["full"])),
        }
    return out


def report(title, res):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"target coverage = {1-ALPHA:.2f}     (class 4 is the rare type)")
    hdr = (f"{'method':<14}{'marginal':>9}{'worst':>8}{'size':>7}{'full%':>7}"
           "   per-class coverage")
    print(hdr); print("-" * 78)
    for m, r in res.items():
        pc = "  ".join(f"{v:.3f}" for v in r["per_class"])
        print(f"{m:<14}{r['marginal']:>9.3f}{r['worst_class']:>8.3f}"
              f"{r['set_size']:>7.2f}{r['full_frac']:>7.2f}   {pc}")


if __name__ == "__main__":
    import argparse, os
    np.seterr(all="ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.npy"))
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    N_TRIALS = args.trials

    print(f"Regime A exact target priors: {np.round(tilted_priors(), 4)}")
    print(f"Regime B target priors:       {np.round(PI_QUERY_B, 4)}")
    print(f"Reference priors:             {np.round(PI_REF, 4)}")

    out = {}
    out["control"] = aggregate("control", use_oracle=True)
    report("CONTROL -- no shift (all methods must hit 0.900)", out["control"])

    out["A_oracle"] = aggregate("A", use_oracle=True)
    report("REGIME A -- pure covariate shift, ORACLE weights (exact sampler)", out["A_oracle"])

    out["A_est"] = aggregate("A", use_oracle=False)
    report("REGIME A -- ESTIMATED weights (cross-fitted, true source labels)", out["A_est"])

    out["B_oracle"] = aggregate("B", use_oracle=True)
    report("REGIME B -- label + within-class covariate shift, ORACLE weights", out["B_oracle"])

    out["B_est"] = aggregate("B", use_oracle=False)
    report("REGIME B -- ESTIMATED weights (cross-fitted, true source labels)", out["B_est"])

    np.save(args.results, out, allow_pickle=True)
    print(f"\nsaved -> {args.results}")
