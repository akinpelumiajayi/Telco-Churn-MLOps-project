"""Append prediction events to BigQuery for dashboards and delayed-label evaluation."""
import os
from datetime import datetime, timezone
from google.cloud import bigquery

def log_predictions(rows, project_id=None, dataset='telco_mlops', table='prediction_events'):
    project_id=project_id or os.environ['GOOGLE_CLOUD_PROJECT']
    table_id=f'{project_id}.{dataset}.{table}'
    client=bigquery.Client(project=project_id)
    now=datetime.now(timezone.utc).isoformat()
    records=[]
    for r in rows:
        records.append({'event_timestamp':r.get('event_timestamp',now),'customer_id':r.get('customer_id'),'model_version':r.get('model_version','unknown'),'churn_probability':float(r['churn_probability']),'threshold':float(r['threshold']),'predicted_churn':int(r['predicted_churn']),'actual_churn':r.get('actual_churn'),'contract':r.get('contract'),'tenure':r.get('tenure'),'monthly_charges':r.get('monthly_charges')})
    errors=client.insert_rows_json(table_id,records)
    if errors: raise RuntimeError(f'BigQuery insert errors: {errors}')
    return len(records)
