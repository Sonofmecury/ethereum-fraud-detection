# Data

Place the **real** labeled Ethereum fraud dataset here as `transaction_dataset.csv`.

Source (the canonical Aliyev "Ethereum Fraud Detection" dataset, ~9,841 addresses, 51 columns, binary `FLAG`):
- Kaggle: `vagifa/ethereum-frauddetection-dataset`
- GitHub mirror: https://github.com/eltontay/Ethereum-Fraud-Detection

The loader (`src/data_loader.py`) auto-detects `data/transaction_dataset.csv`.
If absent, it generates a clearly-labeled **synthetic** dataset with the same
51-column schema so the pipeline is runnable; synthetic results are marked as
such and must NOT be reported as real. Drop the real CSV in and re-run to get
real-data results with no code changes.
