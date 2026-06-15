"""
data_loader.py -- load the real Ethereum fraud dataset if present, else generate
a schema-faithful SYNTHETIC dataset so the pipeline is runnable.

Real data: data/transaction_dataset.csv (Aliyev dataset, 51 cols incl. FLAG).
If absent, synthesize the same 51-column schema with a realistic class imbalance
(~22% fraud, matching the published dataset) and fraud/benign differences spread
across all three feature classes, so the feature-class comparison is exercised.

`load()` returns (DataFrame, source_str) where source_str is "real" or
"synthetic". The manuscript MUST report which was used.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import feature_classes as fc

HERE = os.path.dirname(__file__)
REAL_CSV = os.path.join(HERE, "..", "data", "transaction_dataset.csv")

# Canonical column order of the Aliyev dataset (51 columns).
COLUMNS = ["", "Index", "Address", "FLAG",
    "Avg min between sent tnx", "Avg min between received tnx",
    "Time Diff between first and last (Mins)", "Sent tnx", "Received Tnx",
    "Number of Created Contracts", "Unique Received From Addresses",
    "Unique Sent To Addresses", "min value received", "max value received ",
    "avg val received", "min val sent", "max val sent", "avg val sent",
    "min value sent to contract", "max val sent to contract",
    "avg value sent to contract",
    "total transactions (including tnx to create contract",
    "total Ether sent", "total ether received", "total ether sent contracts",
    "total ether balance", " Total ERC20 tnxs", " ERC20 total Ether received",
    " ERC20 total ether sent", " ERC20 total Ether sent contract",
    " ERC20 uniq sent addr", " ERC20 uniq rec addr", " ERC20 uniq sent addr.1",
    " ERC20 uniq rec contract addr", " ERC20 avg time between sent tnx",
    " ERC20 avg time between rec tnx", " ERC20 avg time between rec 2 tnx",
    " ERC20 avg time between contract tnx", " ERC20 min val rec",
    " ERC20 max val rec", " ERC20 avg val rec", " ERC20 min val sent",
    " ERC20 max val sent", " ERC20 avg val sent", " ERC20 min val sent contract",
    " ERC20 max val sent contract", " ERC20 avg val sent contract",
    " ERC20 uniq sent token name", " ERC20 uniq rec token name",
    " ERC20 most sent token type", " ERC20_most_rec_token_type"]


def _synthesize(n=9841, fraud_frac=0.221, seed=42):
    """Schema-faithful synthetic data. Fraud accounts differ across ALL three
    feature classes (lower counterparty diversity, burstier timing, different
    value profile) with heavy overlap + noise so the task is non-trivial."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_frac).astype(int)
    df = pd.DataFrame({"": np.arange(n), "Index": np.arange(1, n + 1),
                       "Address": ["0x%040x" % i for i in range(n)], "FLAG": y})

    def feat(benign_mean, fraud_mean, spread, nonneg=True):
        base = np.where(y == 1, fraud_mean, benign_mean)
        v = base * np.exp(rng.normal(0, spread, n))  # log-normal-ish, skewed
        return np.clip(v, 0, None) if nonneg else v

    resolved_like = {  # use canonical (spaced) names from COLUMNS
        "graph": [c for c in COLUMNS if fc._norm(c) in {fc._norm(x) for x in fc.GRAPH}],
        "transaction": [c for c in COLUMNS if fc._norm(c) in {fc._norm(x) for x in fc.TRANSACTION}],
        "temporal": [c for c in COLUMNS if fc._norm(c) in {fc._norm(x) for x in fc.TEMPORAL}],
    }
    # GRAPH: fraud interacts with fewer distinct counterparties/tokens
    for c in resolved_like["graph"]:
        df[c] = feat(benign_mean=18, fraud_mean=6, spread=0.9).round()
    # TRANSACTION: fraud moves different value/volumes
    for c in resolved_like["transaction"]:
        df[c] = feat(benign_mean=50, fraud_mean=120, spread=1.2)
    # TEMPORAL: fraud is burstier (shorter gaps), shorter lifetime
    for c in resolved_like["temporal"]:
        df[c] = feat(benign_mean=8000, fraud_mean=1500, spread=1.0)
    # categorical / dropped text columns
    df[" ERC20 most sent token type"] = "None"
    df[" ERC20_most_rec_token_type"] = "None"
    # ensure all schema columns exist, ordered
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = 0.0
    return df[COLUMNS]


def load():
    if os.path.exists(REAL_CSV):
        df = pd.read_csv(REAL_CSV)
        return df, "real"
    return _synthesize(), "synthetic"


def prepare_xy(df):
    """Return (X DataFrame of numeric features, y array, resolved-classes dict)."""
    resolved = fc.resolve(df.columns)
    feat_cols = fc.all_feature_columns(resolved)
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[fc.LABEL], errors="coerce").fillna(0).astype(int).values
    return X, y, resolved


if __name__ == "__main__":
    d, src = load()
    print("source:", src, "shape:", d.shape, "fraud rate:",
          round(pd.to_numeric(d["FLAG"]).mean(), 3))
