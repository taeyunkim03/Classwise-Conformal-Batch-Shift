"""Regression tests for conformal.py (11 checks). Run: python tests_conformal.py"""
import numpy as np
from conformal import (weighted_threshold, combined_conformal,
                       classwise_conformal, naive_conformal, aps_score_matrix)

rng = np.random.default_rng(0)
K = 4

def expect_error(fn, desc):
    try:
        fn(); raise AssertionError(f"no error raised: {desc}")
    except ValueError:
        print(f"ok  {desc}")

# 1. unit-weight identity: reduces to ceil((n+1)(1-a))-th smallest score
s = rng.uniform(size=100); a = 0.2
t = weighted_threshold(s, np.ones(100), np.ones(1), a)[0]
manual = np.sort(s)[int(np.ceil((101)*(1-a))) - 1]
assert np.isclose(t, manual), (t, manual)
print("ok  unit-weight identity with classic split-conformal quantile")

# 2. shared scaling invariance (label-factor cancellation mechanism)
y = rng.choice(K, size=300); sc = rng.uniform(size=300)
sq = rng.uniform(size=(80, K))
wkc = rng.uniform(0.5, 2.0, size=(300, K)); wkq = rng.uniform(0.5, 2.0, size=(80, K))
base = combined_conformal(sc, y, wkc, sq, wkq, 0.1, K)
c = rng.uniform(0.01, 100, size=K)
same = combined_conformal(sc, y, wkc*c, sq, wkq*c, 0.1, K)
assert np.array_equal(base, same)
print("ok  shared per-class scaling leaves sets identical")

# 3. separate cal/test scaling sensitivity (a scaling error must be visible)
t1 = weighted_threshold(s, np.ones(100), np.ones(1), a)[0]
t2 = weighted_threshold(s, np.ones(100), np.ones(1)*10, a)[0]
assert t1 != t2
print("ok  separate calibration/test scaling changes the threshold")

# 4. missing calibration class -> conservative universal inclusion
y2 = rng.choice([0,1,3], size=200); sc2 = rng.uniform(size=200)
sq2 = rng.uniform(size=(60, K))
inc = combined_conformal(sc2, y2, np.ones((200,K)), sq2, np.ones((60,K)), 0.1, K)
incw = classwise_conformal(sc2, y2, sq2, 0.1, K)
assert inc[:,2].all() and incw[:,2].all() and np.array_equal(inc[:,2], incw[:,2])
print("ok  missing class conservatively included; matches classwise")

# 5-9. input validation
expect_error(lambda: weighted_threshold(s[:2], np.r_[-1.,1.], np.r_[1.], .1), "negative weight rejected")
expect_error(lambda: weighted_threshold(s[:2], np.r_[np.inf,1.], np.r_[1.], .1), "nonfinite weight rejected")
expect_error(lambda: weighted_threshold(s[:3], np.r_[1.,1.], np.r_[1.], .1), "mismatched lengths rejected")
expect_error(lambda: weighted_threshold(s[:2], np.r_[1.,1.], np.r_[1.], 1.5), "invalid alpha rejected")
expect_error(lambda: weighted_threshold(s[:2], np.r_[0.,0.], np.r_[1.], .1), "zero total mass rejected")

# 10. empty calibration -> +inf thresholds (honest abstention)
t = weighted_threshold(np.array([]), np.array([]), np.ones(3), .1)
assert np.all(np.isinf(t))
print("ok  empty calibration yields +inf thresholds")

# 11. probability-column alignment
from sklearn.linear_model import LogisticRegression
from conformal import align_proba
X = rng.normal(size=(120, 5)); yy = rng.choice([0, 2], size=120)
clf = LogisticRegression(max_iter=500).fit(X, yy)
P = align_proba(clf, X[:10], 4)
assert P.shape == (10, 4) and np.allclose(P[:, [1, 3]], 0) and np.allclose(P.sum(1), 1)
print("ok  align_proba maps partial classes into full K-column simplex")

print("\nALL 11 CHECKS PASS")
