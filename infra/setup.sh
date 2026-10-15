#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:=europe-west2}"
: "${BUCKET:=${PROJECT_ID}-telco-churn}"
: "${REPO:=mlops}"

gcloud config set project "$PROJECT_ID"
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com bigquery.googleapis.com cloudbuild.googleapis.com storage.googleapis.com logging.googleapis.com monitoring.googleapis.com
gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" || true
gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" || true
bq --location="$REGION" mk -d "${PROJECT_ID}:telco_churn" || true
