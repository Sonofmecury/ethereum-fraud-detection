"""Sanity tests: feature-class partition is disjoint and covers the schema;
loader returns the expected schema; X is numeric."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import feature_classes as fc
import data_loader as dl


def test_classes_disjoint():
    g, t, p = set(fc.GRAPH), set(fc.TRANSACTION), set(fc.TEMPORAL)
    assert g.isdisjoint(t) and g.isdisjoint(p) and t.isdisjoint(p)


def test_loader_schema_and_numeric():
    df, src = dl.load()
    assert src in ("real", "synthetic")
    assert "FLAG" in df.columns and len(df) > 1000
    X, y, resolved = dl.prepare_xy(df)
    assert X.shape[0] == len(df)
    assert set(y) <= {0, 1}
    # every declared feature resolved to a real column
    assert len(resolved["graph"]) >= 5
    assert len(resolved["transaction"]) >= 10
    assert len(resolved["temporal"]) >= 3
