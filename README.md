# Telco Customer Churn — Google Cloud MLOps Portfolio Project

End-to-end binary classification project using the public IBM Telco Customer Churn dataset. The project covers data quality, leakage-safe preprocessing, class imbalance, probability calibration/threshold tuning, experiment metadata, model packaging, Vertex AI Model Registry/Endpoint deployment, monitoring for drift and classification quality, and CI/CD.

> **Dataset note:** the commonly used IBM Telco Customer Churn sample describes a fictional telecommunications company. It is a real, public dataset/sample, not confidential production data.

## Architecture

```text
IBM Telco CSV
    |
    v
Cloud Storage / BigQuery
    |
    v
Vertex AI Pipeline
    +--> validate
    +--> preprocess
    +--> train (class_weight=balanced)
    +--> tune threshold on validation set
    +--> evaluate on untouched test set
    +--> quality gate
    +--> upload model to Model Registry
    +--> deploy to Vertex AI Endpoint
    |
    v
Online predictions
    |
    +--> request logs / BigQuery
    +--> drift monitor (PSI + categorical distribution checks)
    +--> precision/recall monitor when labels arrive
    |
    v
Cloud Scheduler / pipeline rerun
```

## Business framing

Predict whether a customer will churn in the next period so a retention team can prioritize outreach. The model is **decision support**, not an automatic customer-action system.

Primary metrics:
- PR-AUC: useful under class imbalance
- Recall: how many churners we catch
- Precision: how many flagged customers really churn
- F1: balance between precision and recall
- ROC-AUC: secondary ranking metric

The production decision threshold is tuned on validation data rather than assuming 0.50.

## Dataset

Source: IBM Telco Customer Churn sample. The repository includes a downloader rather than committing the CSV. The public sample is available from IBM documentation and IBM's sample-data mirror.

## Repository

```text
.
├── src/
│   ├── config.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   └── monitor.py
├── pipeline/
│   └── pipeline.py
├── serving/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── infra/
│   ├── setup.sh
│   └── deploy.sh
├── tests/
│   └── test_model.py
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
```

## 1. Local setup

```bash
git clone <your-repo-url>
cd telco-churn-mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset:

```bash
python -m src.data
```

Train and evaluate locally:

```bash
python -m src.train
```

The run writes artifacts to `artifacts/` including:

```text
model.joblib
metrics.json
threshold.json
feature_schema.json
```

Run tests:

```bash
pytest -q
```

## 2. What the training pipeline does

1. Normalizes column names and fixes `TotalCharges` numeric parsing.
2. Removes `customerID` from model features.
3. Splits train/validation/test with stratification.
4. Fits preprocessing **inside** the training split to prevent leakage.
5. One-hot encodes categorical variables and imputes numeric values.
6. Uses logistic regression with `class_weight="balanced"`.
7. Selects a probability threshold on validation data using F1, with a configurable minimum recall.
8. Evaluates once on the untouched test set.
9. Saves one deployable sklearn Pipeline plus the threshold and feature contract.

## 3. Google Cloud setup

Set variables:

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="europe-west2"
export BUCKET="${PROJECT_ID}-telco-churn"
export REPO="mlops"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/telco-churn-serving:latest"

gcloud config set project "$PROJECT_ID"
```

Enable services:

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com
```

Create storage and Artifact Registry:

```bash
gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}"
g
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION"
```

Upload the CSV to Cloud Storage after running `python -m src.data`:

```bash
gcloud storage cp data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  "gs://${BUCKET}/raw/telco.csv"
```

## 4. BigQuery

Create a dataset:

```bash
bq --location="${REGION}" mk -d "${PROJECT_ID}:telco_churn"
```

For a portfolio project, load the raw CSV into BigQuery and keep the raw table immutable. Create curated train/validation/test tables from SQL or from the Python pipeline, but record the split seed and dataset version.

## 5. Build the serving image

The serving image accepts key-value JSON objects so monitoring can associate feature names with values.

```bash
bash infra/build_serving.sh
```

The current Google Cloud documentation lists scikit-learn 1.6 prebuilt prediction images, but this project uses a custom container because it needs custom JSON-to-DataFrame handling and returns both probability and business-facing classification outputs.

## 6. Upload the model to Vertex AI Model Registry

After local or Vertex training, copy the artifact:

```bash
gcloud storage cp artifacts/model.joblib \
  "gs://${BUCKET}/models/v1/model.joblib"
```

Then run:

```bash
python -m infra.deploy \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --bucket "$BUCKET" \
  --image "$IMAGE"
```

The script uploads the model with the custom serving container, creates an endpoint if necessary, and deploys the model.

## 7. Test the endpoint

Example payload:

```json
{
  "instances": [
    {
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 1,
      "PhoneService": "No",
      "MultipleLines": "No phone service",
      "InternetService": "DSL",
      "OnlineSecurity": "No",
      "OnlineBackup": "Yes",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 29.85,
      "TotalCharges": 29.85
    }
  ]
}
```

The response contains:

```json
{
  "predictions": [
    {
      "churn_probability": 0.71,
      "churn_prediction": 1,
      "threshold": 0.42
    }
  ]
}
```

## 8. Monitoring strategy

### Feature drift

Compare production inputs with the training baseline. The included `src.monitor` script supports:
- PSI for numeric features
- total variation distance for categorical features
- missing-value-rate changes

Suggested alert policy:
- PSI < 0.10: no action
- 0.10–0.20: investigate
- > 0.20: trigger a review/retraining candidate

These are engineering heuristics, not universal statistical laws; tune them to the business and data volume.

### Classification quality

Feature drift alone does not prove model failure. Once delayed churn labels arrive, calculate:
- precision
- recall
- F1
- PR-AUC
- confusion matrix

Store predictions and eventual labels in BigQuery. A scheduled monitoring job can alert when recall or precision falls below the agreed service-level threshold.

### Vertex AI Model Monitoring

For a managed Google Cloud implementation, register the model and configure Model Monitoring with a feature schema and training baseline. Current Google Cloud documentation describes monitoring of categorical/numerical feature drift and prediction-output drift, with scheduled or on-demand monitoring jobs.

## 9. CI/CD

The GitHub Actions workflow runs:
- formatting/lint checks
- unit tests
- local training smoke test

For production, add a protected deployment environment that authenticates to Google Cloud using Workload Identity Federation rather than a long-lived JSON service-account key. The deployment job should require the model quality gate to pass.

## 10. Portfolio talking points

Be ready to explain:

**Why class_weight=balanced?** Churn is the minority class; unweighted accuracy can hide poor minority-class recall.

**Why PR-AUC?** Precision-recall performance is often more informative than ROC-AUC when the positive class is relatively uncommon.

**Why tune the threshold?** 0.5 is not a business rule. The threshold controls the precision/recall trade-off.

**Why fit preprocessing only on training data?** To avoid leaking validation/test distribution information into training.

**Why monitor features and labels separately?** Drift can occur without immediate performance degradation, while actual performance requires ground-truth labels that often arrive later.

**Why keep the model and threshold together?** A model's probability output is not the same thing as the operational classification policy. The deployed artifact must preserve the exact threshold used for the decision.

**Why Model Registry?** It provides versioned model lineage and a controlled path from candidate to deployment.

## 11. Cost hygiene

For a portfolio project, destroy endpoints when you are finished. Online prediction endpoints can incur charges while deployed. Prefer local training and small CPU resources for experimentation, then use Vertex AI for the production workflow demonstration.

## References

- IBM Telco Customer Churn sample
- Google Cloud Vertex AI / Model Registry / Pipelines / Model Monitoring documentation
```

cat > requirements.txt <<'EOF'
pandas>=2.2,<3
numpy>=1.26,<3
scikit-learn>=1.6,<1.7
joblib>=1.4,<2
imbalanced-learn>=0.12,<0.14
pyyaml>=6,<7
pytest>=8,<9
google-cloud-aiplatform>=1.100,<2
google-cloud-storage>=2.18,<3
google-cloud-bigquery>=3.30,<4
kfp>=2.8,<3

## Portfolio extensions: experiments, explainability, logging, scheduled retraining, dashboard

This repository now includes the following working code paths:

- `src/train.py`: trains Logistic Regression and XGBoost with the same leakage-safe preprocessing; tunes threshold on validation data; reports Precision, Recall, F1, PR-AUC, ROC-AUC; logs runs/metrics to MLflow when `MLFLOW_TRACKING_URI` is configured; writes the selected model and SHAP global importance artifacts.
- `pipeline/pipeline.py`: compilable Vertex AI Pipeline component that downloads a versioned CSV from GCS, compares both classifiers, logs metrics to Vertex AI Experiments, and uploads the model, metrics, and SHAP feature importance to GCS.
- `src/prediction_logger.py`: append prediction events to BigQuery for downstream monitoring.
- `dashboard/create_views.sql`: BigQuery tables/views suitable as Looker Studio data sources.
- `scheduler/`: Cloud Run HTTP service to submit a new Vertex AI Pipeline run, suitable for a Cloud Scheduler HTTP trigger.

### A. Dataset and local experiment comparison

Use the IBM Telco Customer Churn CSV (commonly named `WA_Fn-UseC_-Telco-Customer-Churn.csv`). Download it from the public IBM sample-data source or the source link in the project introduction and put it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. Do not commit customer-level data to Git. The project is a public, fictional-company sample dataset; do not treat it as a production customer dataset.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.train
python -m pytest -q
```

Artifacts are written to `artifacts/`: `model.joblib`, `metrics.json`, and (if SHAP succeeds) `shap_global_importance.csv` and `shap_summary.png`. `xgboost` and `shap` are explicit dependencies. SHAP can be computationally heavy; explanations use a capped sample. For a strictly unbiased model comparison, choose the winning candidate using validation PR-AUC only, then evaluate the chosen candidate once on the holdout test set. The pipeline emits holdout metrics for both candidates as a portfolio comparison, so don't use the holdout results repeatedly to tune hyperparameters.

### B. Experiment tracking

For durable shared tracking, configure MLflow with a tracking server and artifact store accessible to the training environment. For a quick local experiment log, leave `MLFLOW_TRACKING_URI` unset and the script will still write metric JSON files; for MLflow logging, set for example:

```bash
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_TRACKING_SERVER:5000
export MLFLOW_EXPERIMENT=telco-churn-model-comparison
python -m src.train
```

The managed Vertex AI pipeline uses Vertex AI Experiments (`aiplatform.init(..., experiment=...)`, `start_run`, `log_params`, `log_metrics`) so the cloud runs and model candidates can be compared from the Vertex AI console. The training code records the chosen threshold alongside the fitted preprocessing+classifier pipeline; never deploy a threshold different from the one evaluated.

### C. Upload the dataset to GCS and compile/submit the real Vertex AI Pipeline

```bash
export PROJECT_ID="$(gcloud config get-value project)"
export REGION=europe-west2
export BUCKET="${PROJECT_ID}-mlops"
gcloud storage cp data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  "gs://${BUCKET}/data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv"
python -m pipeline.pipeline
```

Upload the compiled template for scheduled runs, and then submit it (the pipeline uses the same parameters shown below):

```bash
gcloud storage cp telco_churn_pipeline.json "gs://${BUCKET}/pipelines/telco_churn_pipeline.json"
```


```python
from google.cloud import aiplatform
project_id = "YOUR_PROJECT_ID"
region = "europe-west2"
aiplatform.init(project=project_id, location=region)
job = aiplatform.PipelineJob(
    display_name="telco-churn-model-comparison",
    template_path="telco_churn_pipeline.json",
    parameter_values={
        "data_gcs_uri": f"gs://{project_id}-mlops/data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "artifact_gcs_prefix": f"gs://{project_id}-mlops/artifacts/telco-churn/run-001",
        "project_id": project_id,
        "experiment_name": "telco-churn",
    },
    enable_caching=False,
)
job.run()
```

Run this with an authenticated principal/service account that has permission to read the input GCS object, write artifacts, create Vertex AI pipeline jobs, and write Vertex AI Experiments. Enable `aiplatform.googleapis.com`, `storage.googleapis.com`, and `iam.googleapis.com`. For repeat runs, use a unique artifact prefix per run. Pipeline compilation requires the `kfp` dependency.

### D. BigQuery prediction logging

Create the dataset/table/views in `dashboard/create_views.sql`, replacing `PROJECT_ID` with your actual project ID. The service account writing events needs BigQuery Job User (if querying) and BigQuery Data Editor on the target dataset. Example call:

```python
from src.prediction_logger import log_predictions
log_predictions([{
  "customer_id": "demo-001",
  "model_version": "telco-churn-2026-10-01",
  "churn_probability": 0.71,
  "threshold": 0.42,
  "predicted_churn": 1,
  "actual_churn": None,  # populate later when the label is known
  "contract": "Month-to-month",
  "tenure": 1,
  "monthly_charges": 29.85,
}], project_id="YOUR_PROJECT_ID")
```

Never log direct identifiers or personal data unless your privacy/security policy explicitly allows it. `customer_id` is optional and should normally be a pseudonymous key. Delayed labels are important: when churn outcomes arrive, update the corresponding event's `actual_churn` value (or write a separate labels table and join it in a view). Don't evaluate precision/recall on unlabeled predictions.

### E. Looker Studio monitoring dashboard

1. Open Looker Studio and create a report.
2. Add a BigQuery data source for `PROJECT_ID.telco_mlops.vw_churn_overview`.
3. Add a time-series chart: `event_date` dimension, `predicted_churn_rate` and `avg_churn_probability` metrics.
4. Add a scorecard for `prediction_count` and a time series for `observed_churn_rate` once delayed labels arrive.
5. Add `PROJECT_ID.telco_mlops.vw_model_performance` as another source; graph precision and recall by `event_date`, with `model_version` as a breakdown dimension.
6. Add `PROJECT_ID.telco_mlops.vw_churn_segments`; chart average churn probability by contract type and date.
7. Add date-range and model-version filters. Display a note that metrics based on actual labels lag prediction time.

This dashboard is designed as a live BigQuery-backed report. It is not a pre-rendered dashboard file; Looker Studio reports are created in the user's Google account and permissions. Grant report viewers only the dataset access they need; decide between owner credentials and viewer credentials according to your organization's governance.

### F. Scheduled retraining with Cloud Scheduler

The scheduler service is intentionally a thin authenticated trigger: it starts a new Vertex AI Pipeline job; it does not itself decide whether to promote a model. Deploy `scheduler/` to Cloud Run with environment variables:

- `PROJECT_ID`, `REGION`
- `PIPELINE_TEMPLATE_URI` (GCS URI to the compiled `telco_churn_pipeline.json`)
- `DATA_GCS_URI` (GCS URI to the current approved training dataset)
- `ARTIFACT_GCS_PREFIX` (unique run prefix, or use a timestamped prefix strategy)
- `SCHEDULER_TOKEN` (strong random secret; alternatively use authenticated Cloud Scheduler/OIDC and remove the shared-token check)

Build/deploy from the repository root after enabling Cloud Run, Cloud Build, Artifact Registry and Scheduler APIs. Example outline:

```bash
gcloud builds submit scheduler --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/mlops/churn-scheduler:latest"
gcloud run deploy churn-retraining-trigger \
  --image "${REGION}-docker.pkg.dev/${PROJECT_ID}/mlops/churn-scheduler:latest" \
  --region "$REGION" --no-allow-unauthenticated \
  --set-env-vars "PROJECT_ID=${PROJECT_ID},REGION=${REGION},PIPELINE_TEMPLATE_URI=gs://${BUCKET}/pipelines/telco_churn_pipeline.json,DATA_GCS_URI=gs://${BUCKET}/data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv,ARTIFACT_GCS_PREFIX=gs://${BUCKET}/artifacts/telco-churn/scheduled-run,SCHEDULER_TOKEN=REPLACE_WITH_SECRET"
```

For production, configure a dedicated Cloud Run runtime service account with least privilege and use Cloud Scheduler OIDC authentication instead of a static token. Give that identity `roles/run.invoker` on the service, and give the Cloud Run runtime identity permissions to create Vertex AI Pipeline jobs, read the template/data, and write artifacts. Create a schedule (example daily at 03:00 UTC):

```bash
gcloud scheduler jobs create http telco-churn-retraining-daily \
  --location "$REGION" --schedule="0 3 * * *" --time-zone="Etc/UTC" \
  --uri="YOUR_CLOUD_RUN_URL/run" --http-method=POST \
  --update-headers="X-Scheduler-Token=REPLACE_WITH_SECRET"
```

For an enterprise setup, prefer Scheduler OIDC rather than a shared header secret, use Secret Manager for secrets, add alerting on failed pipeline runs, and implement a promotion stage that registers/deploys only if quality and governance gates pass. The sample trigger starts retraining; it does not automatically deploy every model.

### G. Explainability and responsible use

SHAP global importance is emitted to `shap_global_importance.csv`; local explanations can be added for a specific scored customer. One-hot encoded features may appear with transformed feature names. SHAP values explain the model's contribution patterns, not causal reasons why an individual customer will churn. Review for leakage, bias, stability, and business validity before acting on scores.

### Cloud cost and teardown

Vertex AI endpoints, pipeline runs, Cloud Run, BigQuery storage/queries, and artifact storage can incur charges. For a learning project, run training locally first, deploy an endpoint only when needed, and delete endpoints/jobs/resources after the exercise. Check current regional pricing and service quotas before running cloud workloads.
