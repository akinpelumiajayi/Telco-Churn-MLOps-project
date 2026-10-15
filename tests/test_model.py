from pathlib import Path
import pandas as pd
from src.train import clean, make_pipeline


def test_clean_maps_target_and_numeric():
    raw = pd.DataFrame({
        "TotalCharges": ["10.5", ""],
        "Churn": ["Yes", "No"],
        "x": [1, 2],
    })
    out = clean(raw)
    assert out["Churn"].tolist() == [1, 0]
    assert str(out["TotalCharges"].dtype).startswith("float")


def test_pipeline_can_fit_small_sample():
    df = pd.DataFrame({
        "num": [1, 2, 3, 4, 5, 6, 7, 8],
        "cat": ["a", "a", "b", "b", "a", "b", "a", "b"],
    })
    y = [0, 0, 0, 1, 1, 1, 0, 1]
    p = make_pipeline(df)
    p.fit(df, y)
    assert len(p.predict_proba(df)) == 8
