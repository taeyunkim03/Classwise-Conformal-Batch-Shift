"""
Core conformal prediction machinery for cell-type annotation under distribution shift.

Implements:
  - APS (Adaptive Prediction Sets) nonconformity scores, randomized and not
  - Weighted-quantile threshold (Tibshirani et al. 2019 construction)
  - Four prediction-set methods:
      1. naive        : marginal split conformal
      2. classwise    : Mondrian / class-conditional split conformal
      3. weighted     : marginal weighted conformal (covariate shift)
      4. combined     : classwise + within-class weighting

Design note on method 4 (the proposed method)
---------------------------------------------
Under a Mondrian partition by true class, the calibration points in cell k are
drawn from p(x | Y=k) and the query points in cell k from q(x | Y=k). The
label-proportion factor q(k)/p(k) does NOT appear, because conditioning on the
class absorbs it -- equivalently, it cancels when weights are normalized within
the cell. So the combined correction is exactly:

    Mondrian partition by class  (handles label shift, structurally)
      + within-cell weighting by q(x|k)/p(x|k)  (handles covariate shift)

This needs no invariance assumption on P(Y) and none on P(Y|X); it only
requires estimating the within-class density ratio.
"""

import numpy as np


# --------------------------------------------------------------------------
# Nonconformity scores
# --------------------------------------------------------------------------

def aps_score_matrix(probs, randomized=True, rng=None):
    """
    APS score for every candidate label (Romano, Sesia & Candes 2020).

    s(x, y) = sum of predicted probabilities of all labels ranked at or above y.
    Randomized version subtracts U * p_y with a single U per sample, which
    yields exact (rather than conservative) coverage under exchangeability.

    Parameters
    ----------
    probs : (n, K) array of predicted class probabilities (rows sum to 1)

    Returns
    -------
    (n, K) array of scores in the ORIGINAL label ordering. Lower = more conforming.
    """
    probs = np.asarray(probs, dtype=float)
    n, K = probs.shape

    order = np.argsort(-probs, axis=1)                  # descending
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)                   # inclusive cumulative mass

    if randomized:
        if rng is None:
            rng = np.random.default_rng()
        u = rng.uniform(size=(n, 1))
        sorted_scores = cum - u * sorted_p
    else:
        sorted_scores = cum

    # scatter back to original label positions
    scores = np.empty_like(sorted_scores)
    np.put_along_axis(scores, order, sorted_scores, axis=1)
    return scores


def true_label_scores(score_matrix, labels):
    """Pull out s(x_i, y_i) for the observed labels."""
    return score_matrix[np.arange(len(labels)), labels]


# --------------------------------------------------------------------------
# Weighted quantile threshold
# --------------------------------------------------------------------------

def weighted_threshold(cal_scores, cal_weights, test_weights, alpha):
    """
    Tibshirani et al. (2019) weighted conformal threshold.

    Finds the smallest t such that

        sum_{i: s_i <= t} w_i  >=  (1 - alpha) * ( sum_i w_i + w_test )

    The test point contributes its own weight to the normalization, which is what
    makes the guarantee finite-sample rather than asymptotic. If no such t exists
    (the test weight is large relative to the calibration mass) the threshold is
    +inf, i.e. the prediction set is all labels -- an honest abstention.

    Setting all weights to 1 recovers the standard ceil((n+1)(1-alpha))/n
    split-conformal quantile exactly.

    Parameters
    ----------
    cal_scores  : (n,) calibration scores
    cal_weights : (n,) nonnegative calibration weights
    test_weights: (m,) nonnegative per-test-point weights
    alpha       : miscoverage level

    Returns
    -------
    (m,) array of per-test-point thresholds.
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_weights = np.asarray(cal_weights, dtype=float)
    test_weights = np.asarray(test_weights, dtype=float)

    # --- input validation: silently-invalid weights produce
    # meaningless thresholds that look plausible, which is the failure mode
    # this whole project is about. Fail loudly instead.
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if cal_scores.ndim != 1 or cal_weights.ndim != 1 or test_weights.ndim != 1:
        raise ValueError("cal_scores, cal_weights, test_weights must be 1-D")
    if len(cal_scores) != len(cal_weights):
        raise ValueError(f"length mismatch: {len(cal_scores)} scores vs "
                         f"{len(cal_weights)} weights")
    if not np.all(np.isfinite(cal_scores)):
        raise ValueError("cal_scores contains non-finite values")
    if not (np.all(np.isfinite(cal_weights)) and np.all(np.isfinite(test_weights))):
        raise ValueError("weights contain non-finite values")
    if np.any(cal_weights < 0) or np.any(test_weights < 0):
        raise ValueError("weights must be non-negative")
    if len(cal_weights) > 0 and cal_weights.sum() <= 0:
        raise ValueError("total calibration weight mass must be positive")

    if len(cal_scores) == 0:
        return np.full(len(test_weights), np.inf)

    idx = np.argsort(cal_scores)
    s_sorted = cal_scores[idx]
    w_sorted = cal_weights[idx]

    cumw = np.cumsum(w_sorted)
    total = cumw[-1]

    targets = (1.0 - alpha) * (total + test_weights)
    pos = np.searchsorted(cumw, targets, side="left")

    out = np.full(len(test_weights), np.inf)
    valid = pos < len(s_sorted)
    out[valid] = s_sorted[pos[valid]]
    return out


# --------------------------------------------------------------------------
# The four methods. Each returns a boolean (n_query, K) inclusion mask.
# --------------------------------------------------------------------------

def naive_conformal(cal_scores, query_score_matrix, alpha):
    """Marginal split conformal. Valid only under exchangeability."""
    n_q = query_score_matrix.shape[0]
    w = np.ones(len(cal_scores))
    tau = weighted_threshold(cal_scores, w, np.ones(n_q), alpha)
    return query_score_matrix <= tau[:, None]


def classwise_conformal(cal_scores, cal_labels, query_score_matrix, alpha, K):
    """
    Mondrian / class-conditional split conformal.

    Note: this is ALSO the correct correction for *pure* label shift. Because
    calibration happens within each class, a change in P(Y) between reference and
    query does not disturb validity, provided P(X | Y) is unchanged. This is why
    'classwise' doubles as the label-shift baseline in the comparison.
    """
    n_q = query_score_matrix.shape[0]
    taus = np.full(K, np.inf)
    for k in range(K):
        s_k = cal_scores[cal_labels == k]
        if len(s_k) == 0:
            continue
        taus[k] = weighted_threshold(s_k, np.ones(len(s_k)), np.ones(1), alpha)[0]
    return query_score_matrix <= taus[None, :]


def weighted_conformal(cal_scores, cal_w, query_score_matrix, query_w, alpha):
    """
    Marginal weighted conformal under covariate shift.

    Assumes P(Y | X) is invariant and reweights calibration points by
    w(x) = q(x) / p(x). Restores MARGINAL coverage; says nothing about
    per-class coverage.
    """
    tau = weighted_threshold(cal_scores, cal_w, query_w, alpha)
    return query_score_matrix <= tau[:, None]


def combined_conformal(cal_scores, cal_labels, cal_w_by_class,
                       query_score_matrix, query_w_by_class, alpha, K):
    """
    Proposed method: Mondrian partition by class + within-class weighting.

    cal_w_by_class   : (n_cal, K)   w_k(x_i) = q(x_i | k) / p(x_i | k)
    query_w_by_class : (n_query, K) same ratio evaluated at query points

    For candidate label k, the threshold is calibrated on reference points whose
    true label is k, reweighted toward the query's within-class covariate
    distribution for class k.
    """
    n_q = query_score_matrix.shape[0]
    # Initialize to True: a class with NO calibration examples gets an implicit
    # +inf threshold (conservative inclusion), matching classwise_conformal.
    # Initializing to False would silently EXCLUDE such classes from every set,
    # which is anti-conservative.
    included = np.ones((n_q, K), dtype=bool)
    for k in range(K):
        mask = cal_labels == k
        if mask.sum() == 0:
            continue
        tau_k = weighted_threshold(
            cal_scores[mask], cal_w_by_class[mask, k],
            query_w_by_class[:, k], alpha
        )
        included[:, k] = query_score_matrix[:, k] <= tau_k
    return included


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def align_proba(clf, Z, K):
    """Map clf.predict_proba columns into a full K-column matrix in the GLOBAL
    label order. Classes absent from training get probability 0. Lives here so
    core regression tests do not depend on the single-cell stack."""
    P = np.zeros((Z.shape[0], K))
    P[:, np.asarray(clf.classes_, dtype=int)] = clf.predict_proba(Z)
    return P


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(included, query_labels, K):
    """Marginal coverage, per-class coverage, mean set size."""
    n = len(query_labels)
    covered = included[np.arange(n), query_labels]
    per_class = np.full(K, np.nan)
    for k in range(K):
        m = query_labels == k
        if m.sum() > 0:
            per_class[k] = covered[m].mean()
    sizes = included.sum(axis=1)
    return {
        "marginal": covered.mean(),
        "per_class": per_class,
        "set_size": sizes.mean(),
        "full_frac": float((sizes == K).mean()),
        "empty_frac": float((sizes == 0).mean()),
        "class_counts": np.bincount(query_labels, minlength=K),
    }
