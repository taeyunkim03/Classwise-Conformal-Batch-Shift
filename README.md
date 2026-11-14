# Classwise conformal prediction under batch and technology shift

Cell-type labels are often transferred to new datasets, and the new data usually comes from a different batch or sequencing technology than the reference. Conformal prediction can give each predicted label a coverage guarantee, but only if the new cells resemble the calibration cells. This project asks whether classwise conformal prediction still reaches its target when that assumption is clearly broken.

## Setup

The data is the scIB human pancreas benchmark, with nine batches from six technologies and 13 cell types. For seven of the batches, I held that batch out and used the rest as the reference. The classifier is logistic regression on principal components of highly variable genes, fitted on the training split only. Prediction sets come from classwise (Mondrian) split conformal with APS scores at a 90% target, averaged over 20 random splits per held-out batch.

## Results

| Held-out batch | Domain AUC | Overall coverage | Average, common cell types | Worst common cell type | Score AUC |
|---|---|---|---|---|---|
| celseq | 0.974 | 0.879 | 0.891 | 0.848 | 0.499 |
| celseq2 | 0.998 | 0.889 | 0.887 | 0.809 | 0.506 |
| fluidigmc1 | 0.999 | 0.881 | 0.900 | 0.897 | 0.478 |
| inDrop1 | 0.976 | 0.903 | 0.907 | 0.893 | 0.494 |
| inDrop3 | 0.992 | 0.888 | 0.886 | 0.807 | 0.515 |
| smarter | 0.999 | 0.904 | 0.902 | 0.897 | not run |
| smartseq2 | 0.982 | 0.897 | 0.894 | 0.885 | 0.514 |

"Common" means at least 50 cells in the held-out batch. Domain AUC measures how well a classifier can tell held-out cells from reference cells. Score AUC measures the same thing using only the conformal score, compared within each cell type.

Three things stand out.

1. Every held-out batch was almost perfectly separable from the reference, yet overall coverage stayed near 90%.
2. The scores themselves barely moved, with score AUC near 0.5. Classwise conformal only needs the scores to match within each cell type, which likely explains why coverage held.
3. Individual cell types were less reliable. Ductal cells in celseq, acinar cells in celseq2, and delta cells in inDrop3 fell to between 0.81 and 0.85.

## Other things I tried

Importance-weighted conformal is the usual fix for covariate shift, but the reference and held-out data barely overlap here (relative effective sample size 0.001 to 0.008), so the weights are unusable. Harmony integration raised the effective sample size to 0.14 to 0.23, but weighted conformal still did not clearly beat classwise. The closest prior work (Corbetta et al., listed below) assumes integration has already removed systematic differences within each cell type, but after Harmony, held-out cells could still be partly told apart from reference cells of the same type (AUC 0.64 to 0.81, averaged over cell types).

## Limitations

These results come from one tissue, one classifier, and one score type. Splits are at the cell level rather than the donor level. Only cell types present in both the reference and the held-out batch were kept, so this is a closed-set setting. Treat the numbers as empirical results on this benchmark, not general guarantees.

## Files

| File | What it does |
|---|---|
| `conformal.py` | Conformal methods and APS scores |
| `pancreas_gonogo.py` | Runs one held-out batch end to end |
| `sweep_technologies.sh` | Runs all seven held-out batches |
| `score_shift_diagnostic.py`, `score_shift_sweep.py` | Within-cell-type score comparison |
| `overlap_diagnostic.py` | Checks whether integration removes the shift |
| `rare_class_experiment.py`, `label_budget.py` | Side experiments on rare cell types and labeled query cells |
| `synthetic_validation.py` | Checks the methods on synthetic data where the true answer is known, and saves `results.npy` |
| `make_figure.py` | Plots the synthetic results. Run `synthetic_validation.py` first |
| `tests_conformal.py`, `smoke_test.py` | Checks on the code itself |

## Running it

```bash
conda create -n cpsc python=3.11 -y
conda activate cpsc
pip install -r requirements.txt

python tests_conformal.py
python smoke_test.py
bash fetch_data.sh             # downloads the pancreas data into data/
bash sweep_technologies.sh     # main experiment
python score_shift_sweep.py --data data/pancreas.h5ad
```

Results from my runs are in `sweep/`, `score_shift_sweep.json`, and the `results_*.json` files.

## References

The pancreas data is from Luecken et al., "Benchmarking atlas-level data integration in single-cell genomics," *Nature Methods* (2022).

The closest prior work is Corbetta et al., "Conformal inference for cell type annotation with graph-structured constraints," arXiv:2410.23786.
