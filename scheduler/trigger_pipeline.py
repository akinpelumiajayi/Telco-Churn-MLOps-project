"""Cloud Run HTTP target for Cloud Scheduler. Starts a fresh Vertex AI Pipeline run."""
import os
from flask import Flask, request, jsonify
from google.cloud import aiplatform
app=Flask(__name__)
@app.post('/run')
def run():
    expected=os.environ.get('SCHEDULER_TOKEN')
    if expected and request.headers.get('X-Scheduler-Token') != expected: return jsonify({'error':'unauthorized'}),401
    project=os.environ['PROJECT_ID']; region=os.getenv('REGION','europe-west2')
    template=os.environ['PIPELINE_TEMPLATE_URI']; data_uri=os.environ['DATA_GCS_URI']; artifacts=os.environ['ARTIFACT_GCS_PREFIX']
    aiplatform.init(project=project,location=region)
    job=aiplatform.PipelineJob(display_name='scheduled-telco-churn-retraining',template_path=template,parameter_values={'data_gcs_uri':data_uri,'artifact_gcs_prefix':artifacts,'project_id':project,'experiment_name':'telco-churn'},enable_caching=False)
    job.submit()
    return jsonify({'status':'submitted','pipeline_job_resource_name':job.resource_name}),202
@app.get('/healthz')
def health(): return 'ok',200
