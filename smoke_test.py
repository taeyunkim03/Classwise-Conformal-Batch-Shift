"""Smoke test for pancreas_gonogo.py on synthetic data.

Exercises: donor-aware grouped splitting, a rare class that triggers per-class
weight fallback, classes absent from training splits (align_proba), and the
diagnostic output. Numbers are meaningless (synthetic AnnData); only code
paths and shapes are under test.
"""
import numpy as np, anndata as ad, pandas as pd, sys
import pancreas_gonogo as P

rng = np.random.default_rng(0)
techs  = ["celseq", "celseq2", "inDrop1", "smartseq2"]
ctypes = ["alpha", "beta", "delta", "gamma", "acinar", "ductal", "rare_type"]
n_genes = 400
sig = rng.normal(size=(len(ctypes), n_genes)) * 0.22
batch = rng.normal(size=(len(techs), n_genes)) * 0.18

Xs, ts, cs, ds = [], [], [], []
donor = 0
for ti, t in enumerate(techs):
    for _ in range(4):
        donor += 1
        p = rng.dirichlet(np.r_[np.ones(6) * 2.0, 0.12])
        n = int(rng.integers(220, 420))
        y = rng.choice(len(ctypes), size=n, p=p)
        Xs.append(np.abs(sig[y] + batch[ti] + rng.normal(size=(n, n_genes)) * 0.55))
        ts += [t] * n; cs += [ctypes[k] for k in y]; ds += [f"donor{donor}"] * n

adata = ad.AnnData(X=np.vstack(Xs).astype(np.float32))
adata.obs["tech"] = pd.Categorical(ts)
adata.obs["celltype"] = pd.Categorical(cs)
adata.obs["donor"] = pd.Categorical(ds)
adata.var_names = [f"g{i}" for i in range(n_genes)]
adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]

P.load_data = lambda path: adata          # matches load_data(args.data)

sys.argv = ["x", "--query", "smartseq2", "--n-trials", "3", "--n-hvg", "200",
            "--n-pcs", "20", "--split-key", "donor", "--out", "/tmp/smoke_test.json"]
P.main()
print("\nSMOKE TEST PASSED")
