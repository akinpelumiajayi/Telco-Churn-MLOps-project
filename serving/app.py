import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

MODEL_PATH = os.environ.get("MODEL_PATH", "/model/model.joblib")
artifact = joblib.load(MODEL_PATH)
model = artifact["pipeline"]
threshold = float(artifact["threshold"])
features = artifact["features"]

app = FastAPI(title="Telco Churn Predictor")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    instances = payload.get("instances")
    if not isinstance(instances, list) or not instances:
        raise HTTPException(400, "payload must contain non-empty instances")
    try:
        frame = pd.DataFrame(instances)
        missing = [c for c in features if c not in frame.columns]
        if missing:
            raise HTTPException(400, f"missing features: {missing}")
        frame = frame[features]
        prob = model.predict_proba(frame)[:, 1]
        return {
            "predictions": [
                {
                    "churn_probability": float(p),
                    "churn_prediction": int(p >= threshold),
                    "threshold": threshold,
                }
                for p in prob
            ]
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
