#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:=europe-west2}"
: "${REPO:=mlops}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/telco-churn-serving:latest"
cp artifacts/model.joblib serving/model.joblib
trap 'rm -f serving/model.joblib' EXIT
gcloud builds submit serving --tag "$IMAGE"
echo "$IMAGE"
