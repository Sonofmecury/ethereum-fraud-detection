"""
feature_classes.py -- the three feature classes for the comparison study.

Maps the columns of the Aliyev "Ethereum Fraud Detection" dataset (51 columns)
into three interpretable classes so we can quantify which signal class drives
fraud detection:

  GRAPH        counterparty / degree structure (how many distinct addresses and
               tokens an account interacts with; contracts created)
  TRANSACTION  value & volume (tx counts, ether/ERC20 totals, value min/max/avg)
  TEMPORAL     timing (avg minutes between tx, active lifetime, ERC20 inter-event)

Column names are matched after stripping surrounding whitespace, because the
raw dataset has inconsistent leading/trailing spaces (e.g. ' ERC20 uniq rec addr').
"""
from __future__ import annotations

LABEL = "FLAG"
DROP = ["", "Index", "Address", "ERC20 most sent token type",
        "ERC20_most_rec_token_type"]  # ids + free-text categoricals

GRAPH = [
    "Unique Received From Addresses",
    "Unique Sent To Addresses",
    "Number of Created Contracts",
    "ERC20 uniq sent addr",
    "ERC20 uniq rec addr",
    "ERC20 uniq sent addr.1",
    "ERC20 uniq rec contract addr",
    "ERC20 uniq sent token name",
    "ERC20 uniq rec token name",
]

TRANSACTION = [
    "Sent tnx",
    "Received Tnx",
    "total transactions (including tnx to create contract",
    "total Ether sent",
    "total ether received",
    "total ether sent contracts",
    "total ether balance",
    "min value received",
    "max value received",
    "avg val received",
    "min val sent",
    "max val sent",
    "avg val sent",
    "min value sent to contract",
    "max val sent to contract",
    "avg value sent to contract",
    "Total ERC20 tnxs",
    "ERC20 total Ether received",
    "ERC20 total ether sent",
    "ERC20 total Ether sent contract",
    "ERC20 min val rec",
    "ERC20 max val rec",
    "ERC20 avg val rec",
    "ERC20 min val sent",
    "ERC20 max val sent",
    "ERC20 avg val sent",
    "ERC20 min val sent contract",
    "ERC20 max val sent contract",
    "ERC20 avg val sent contract",
]

TEMPORAL = [
    "Avg min between sent tnx",
    "Avg min between received tnx",
    "Time Diff between first and last (Mins)",
    "ERC20 avg time between sent tnx",
    "ERC20 avg time between rec tnx",
    "ERC20 avg time between rec 2 tnx",
    "ERC20 avg time between contract tnx",
]

CLASSES = {"graph": GRAPH, "transaction": TRANSACTION, "temporal": TEMPORAL}


def _norm(c: str) -> str:
    return c.strip()


def resolve(df_columns):
    """Map each class's canonical names to the actual df columns (whitespace-tolerant)."""
    norm_to_actual = {_norm(c): c for c in df_columns}
    out = {}
    for cls, names in CLASSES.items():
        out[cls] = [norm_to_actual[_norm(n)] for n in names if _norm(n) in norm_to_actual]
    return out


def all_feature_columns(resolved):
    cols = []
    for cls in ("graph", "transaction", "temporal"):
        cols.extend(resolved[cls])
    return cols
