"""Executable Vertex AI Pipeline: load CSV from GCS, compare models, log metrics, upload artifacts."""
from kfp import dsl
from kfp.dsl import Metrics, Output

@dsl.component(base_image='python:3.11', packages_to_install=[
 'pandas>=2.1,<3','numpy>=1.26,<3','scikit-learn>=1.4,<1.7','joblib>=1.3',
 'xgboost>=2,<3','google-cloud-storage>=2.16','google-cloud-aiplatform>=1.60','mlflow>=2.12','shap>=0.45,<1','matplotlib>=3.8'
])
def train_compare_component(data_gcs_uri: str, artifact_gcs_prefix: str, project_id: str, experiment_name: str, metrics: Output[Metrics]):
    import os, json, tempfile
    import numpy as np, pandas as pd, joblib
    from google.cloud import storage, aiplatform
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
    from xgboost import XGBClassifier
    aiplatform.init(project=project_id, location=os.getenv('VERTEX_REGION','europe-west2'), experiment=experiment_name)
    bucket_name, blob_name = data_gcs_uri.replace('gs://','',1).split('/',1)
    local='/tmp/telco.csv'; storage.Client(project=project_id).bucket(bucket_name).blob(blob_name).download_to_filename(local)
    df=pd.read_csv(local); df.columns=[c.strip() for c in df.columns]; df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce'); df['Churn']=df['Churn'].map({'Yes':1,'No':0}).astype(int)
    X=df.drop(columns=['Churn','customerID']); y=df['Churn']; Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.30,stratify=y,random_state=42); Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,stratify=ytmp,random_state=42)
    nums=Xtr.select_dtypes(include=[np.number]).columns.tolist(); cats=Xtr.select_dtypes(exclude=[np.number]).columns.tolist()
    def prep(): return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),cats)])
    ratio=float((ytr==0).sum()/max(1,(ytr==1).sum())); models={'logistic_regression':LogisticRegression(max_iter=2000,class_weight='balanced',solver='liblinear',random_state=42),'xgboost':XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.04,subsample=.85,colsample_bytree=.85,scale_pos_weight=ratio,eval_metric='logloss',random_state=42,n_jobs=4)}
    scores={}; fitted={}
    for name,clf in models.items():
        p=Pipeline([('preprocess',prep()),('classifier',clf)]); p.fit(Xtr,ytr); pv=p.predict_proba(Xv)[:,1]
        # Choose threshold to maximize F1 subject to recall >= 0.70 on validation only.
        choices=[]
        for t in np.linspace(.05,.95,181):
            pred=(pv>=t).astype(int); r=recall_score(yv,pred,zero_division=0); f=f1_score(yv,pred,zero_division=0)
            if r>=.70: choices.append((f,float(t)))
        threshold=max(choices)[1] if choices else .5
        pt=p.predict_proba(Xte)[:,1]; pred=(pt>=threshold).astype(int)
        m={'threshold':threshold,'precision':float(precision_score(yte,pred,zero_division=0)),'recall':float(recall_score(yte,pred,zero_division=0)),'f1':float(f1_score(yte,pred,zero_division=0)),'pr_auc':float(average_precision_score(yte,pt)),'roc_auc':float(roc_auc_score(yte,pt))}
        scores[name]=m; fitted[name]=(p,threshold)
    winner=max(scores,key=lambda k:scores[k]['pr_auc']); p,t=fitted[winner]
    with aiplatform.start_run(run=f'{winner}-{os.getenv("PIPELINE_JOB_ID","run")}'):
        aiplatform.log_params({'winner':winner,'dataset_uri':data_gcs_uri,'threshold':t})
        aiplatform.log_metrics({f'{n}_{k}':float(v) for n,ms in scores.items() for k,v in ms.items()})
    with tempfile.TemporaryDirectory() as td:
        model_path=f'{td}/model.joblib'; joblib.dump({'pipeline':p,'threshold':t,'features':X.columns.tolist(),'target':'Churn','model_name':winner},model_path)
        client=storage.Client(project=project_id); prefix=artifact_gcs_prefix.replace('gs://','',1); bname,base=prefix.split('/',1); bucket=client.bucket(bname)
        bucket.blob(base.rstrip('/')+'/model.joblib').upload_from_filename(model_path)
        bucket.blob(base.rstrip('/')+'/metrics.json').upload_from_string(json.dumps({'winner':winner,'candidates':scores},indent=2),'application/json')
        # SHAP global explanation, written alongside the model. For tree model use TreeExplainer.
        try:
            import shap
            sample=Xte.sample(min(200,len(Xte)),random_state=42); transformed=p.named_steps['preprocess'].transform(sample); clf=p.named_steps['classifier']
            explainer=shap.TreeExplainer(clf) if winner=='xgboost' else shap.LinearExplainer(clf,transformed); sv=explainer.shap_values(transformed)
            imp=np.abs(np.asarray(sv)).mean(axis=0).reshape(-1); names=p.named_steps['preprocess'].get_feature_names_out()
            out=pd.DataFrame({'feature':names,'mean_abs_shap':imp}).sort_values('mean_abs_shap',ascending=False)
            out.to_csv(f'{td}/shap_global_importance.csv',index=False); bucket.blob(base.rstrip('/')+'/shap_global_importance.csv').upload_from_filename(f'{td}/shap_global_importance.csv')
        except Exception as exc: bucket.blob(base.rstrip('/')+'/shap_error.txt').upload_from_string(str(exc))
    for model_name, model_metrics in scores.items():
        for key,value in model_metrics.items(): metrics.log_metric(f'{model_name}_{key}',float(value))
    metrics.log_metric('winner_xgboost',1.0 if winner=='xgboost' else 0.0)

@dsl.pipeline(name='telco-churn-train-compare-register-artifacts', description='Train and compare churn classifiers, log Vertex AI experiment metrics, and upload model/SHAP artifacts.')
def telco_churn_pipeline(data_gcs_uri: str, artifact_gcs_prefix: str, project_id: str, experiment_name: str='telco-churn'):
    task=train_compare_component(data_gcs_uri=data_gcs_uri,artifact_gcs_prefix=artifact_gcs_prefix,project_id=project_id,experiment_name=experiment_name)
    task.set_caching_options(False)

if __name__=='__main__':
    from kfp import compiler
    compiler.Compiler().compile(pipeline_func=telco_churn_pipeline,package_path='telco_churn_pipeline.json')
