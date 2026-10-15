"""Lightweight production-drift checks for a portfolio deployment.

Input: a baseline CSV and a production CSV with the same feature columns.
Output: JSON metrics. This deliberately separates drift from true model-quality
monitoring, because labels usually arrive later.
"""
import json
import sys
import numpy as np
import pandas as pd


NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def psi(expected, actual, bins=10):
    expected = pd.to_numeric(expected, errors="coerce").dropna().to_numpy()
    actual = pd.to_numeric(actual, errors="coerce").dropna().to_numpy()
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    e = np.histogram(expected, bins=cuts)[0] / len(expected)
    a = np.histogram(actual, bins=cuts)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def tvd(expected, actual):
    e = expected.fillna("<MISSING>").astype(str).value_counts(normalize=True)
    a = actual.fillna("<MISSING>").astype(str).value_counts(normalize=True)
    idx = e.index.union(a.index)
    return float(0.5 * (e.reindex(idx, fill_value=0) - a.reindex(idx, fill_value=0)).abs().sum())


def run(baseline_path, production_path):
    base = pd.read_csv(baseline_path)
    prod = pd.read_csv(production_path)
    out = {"numeric_psi": {}, "categorical_tvd": {}, "missing_rate_change": {}}
    for c in NUMERIC:
        out["numeric_psi"][c] = psi(base[c], prod[c])
    for c in CATEGORICAL:
        out["categorical_tvd"][c] = tvd(base[c], prod[c])
    for c in NUMERIC + CATEGORICAL:
        out["missing_rate_change"][c] = float(prod[c].isna().mean() - base[c].isna().mean())
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m src.monitor baseline.csv production.csv")
    print(json.dumps(run(sys.argv[1], sys.argv[2]), indent=2))
