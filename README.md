# Ethereum Fraud Detection — Which Feature Class Drives Detection?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20703210.svg)](https://doi.org/10.5281/zenodo.20703210)

**Paper 2 of the "Secure Systems in the Quantum Era" portfolio.** A feature-class
comparison: do *graph* (counterparty structure), *transaction* (value/volume), or
*temporal* (timing) features drive machine-learning detection of fraudulent
Ethereum accounts? Rather than another "Random Forest vs XGBoost" benchmark, the
contribution is a measured answer to **what signal actually matters**.

## Status

- ✅ Pipeline complete and validated (loader, feature-class split, CV, metrics, importance, figures, tests).
- ✅ **Real-data results complete + extended** (9,307 accounts with graph features, 17.7% fraud). Manuscript: `paper/Ethereum_Fraud_Detection_Preprint.pdf`.
- **v3 additions:** PageRank investigation (fraud accounts significantly more central, p~1e-151; not a volume proxy, Spearman 0.19-0.36); ablation showing **topology adds the most on top of transaction features** (PR-AUC +0.031); significance via repeated CV (15 estimates, all p<1e-6).
- **Headline:** *true* graph-topology features (PageRank, k-core, clustering, reconstructed from a 242k-node / 1.65M-edge tx graph) significantly beat degree-count features (PR-AUC 0.84 vs 0.70; +0.072 F1, p<0.001), and **PageRank is the single most important feature**. Transaction-value features remain the strongest single class (PR-AUC 0.93, 36% importance); combined is best (PR-AUC 0.979). Three models (LogReg, RandomForest, HistGradientBoosting); paired significance tests; permutation-importance attribution.

## The three feature classes (mapped to the Aliyev dataset)

- **Graph:** unique sent-to / received-from addresses, contracts created, unique ERC20 counterparties and token diversity.
- **Transaction:** tx counts, ether/ERC20 sent-received totals, value min/max/avg.
- **Temporal:** avg minutes between sent/received tx, active lifetime, ERC20 inter-event times.

## Method

Stratified 3-fold CV; class-weighted Logistic Regression and Random Forest;
imbalance-aware metrics (precision, recall, F1, ROC-AUC, PR-AUC) for the fraud
class; per-feature-class models vs combined; Random-Forest importance aggregated
by class.

## Run

```bash
pip install -r requirements.txt
# (optional but recommended) put the real CSV at data/transaction_dataset.csv
bash run_all.sh
pytest -q
```

Outputs: `results/feature_class_results.csv`, `results/feature_importance.csv`,
`results/run_meta.json` (records data source), and `results/figures/`.

## License
MIT.
