-- Create the base table once in BigQuery with columns matching src/prediction_logger.py.
CREATE SCHEMA IF NOT EXISTS `PROJECT_ID.telco_mlops` OPTIONS(location="europe-west2");
CREATE TABLE IF NOT EXISTS `PROJECT_ID.telco_mlops.prediction_events` (
 event_timestamp TIMESTAMP, customer_id STRING, model_version STRING,
 churn_probability FLOAT64, threshold FLOAT64, predicted_churn INT64,
 actual_churn INT64, contract STRING, tenure FLOAT64, monthly_charges FLOAT64
);
CREATE OR REPLACE VIEW `PROJECT_ID.telco_mlops.vw_churn_overview` AS
SELECT DATE(event_timestamp) event_date, COUNT(*) prediction_count,
 AVG(churn_probability) avg_churn_probability, AVG(predicted_churn) predicted_churn_rate,
 AVG(IF(actual_churn IS NOT NULL, CAST(actual_churn AS FLOAT64), NULL)) observed_churn_rate
FROM `PROJECT_ID.telco_mlops.prediction_events` GROUP BY event_date;
CREATE OR REPLACE VIEW `PROJECT_ID.telco_mlops.vw_model_performance` AS
SELECT DATE(event_timestamp) event_date, model_version,
 SAFE_DIVIDE(COUNTIF(predicted_churn=1 AND actual_churn=1), COUNTIF(predicted_churn=1)) precision,
 SAFE_DIVIDE(COUNTIF(predicted_churn=1 AND actual_churn=1), COUNTIF(actual_churn=1)) recall,
 COUNTIF(actual_churn IS NOT NULL) labeled_rows
FROM `PROJECT_ID.telco_mlops.prediction_events`
WHERE actual_churn IS NOT NULL GROUP BY event_date, model_version;
CREATE OR REPLACE VIEW `PROJECT_ID.telco_mlops.vw_churn_segments` AS
SELECT DATE(event_timestamp) event_date, COALESCE(contract,'(unknown)') contract,
 COUNT(*) prediction_count, AVG(churn_probability) avg_churn_probability,
 AVG(predicted_churn) predicted_churn_rate
FROM `PROJECT_ID.telco_mlops.prediction_events` GROUP BY event_date, contract;
