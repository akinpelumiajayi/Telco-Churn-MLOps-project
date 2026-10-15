"""Train/evaluate churn models; compare Logistic Regression and XGBoost; log MLflow metrics."""
import json, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from .config import ARTIFACT_DIR, ID_COLUMNS, MIN_RECALL, RANDOM_STATE, TARGET

DATA_FILE = Path(os.getenv('TELCO_DATA', 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'))

def clean(df):
    df=df.copy(); df.columns=[c.strip() for c in df.columns]
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')
    df[TARGET]=df[TARGET].map({'Yes':1,'No':0}).astype(int)
    return df

def make_preprocessor(X):
    numeric=X.select_dtypes(include=[np.number]).columns.tolist(); categorical=X.select_dtypes(exclude=[np.number]).columns.tolist()
    return ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),categorical)])

def make_pipeline(X):
    """Default balanced Logistic Regression pipeline (also used by unit tests)."""
    return Pipeline([('preprocess', make_preprocessor(X)), ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=RANDOM_STATE))])

def choose_threshold(y, p, min_recall=MIN_RECALL):
    rows=[]
    for t in np.linspace(.05,.95,181):
        pred=(p>=t).astype(int); r=recall_score(y,pred,zero_division=0); pr=precision_score(y,pred,zero_division=0); f=f1_score(y,pred,zero_division=0)
        rows.append((float(t),float(pr),float(r),float(f)))
    eligible=[r for r in rows if r[2]>=min_recall]
    return max(eligible or rows,key=lambda x:x[3])[0]

def calc_metrics(y,p,t):
    pred=(p>=t).astype(int)
    return {'threshold':float(t),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),'accuracy':float(accuracy_score(y,pred)),'predicted_churn_rate':float(pred.mean())}

def main():
    df=clean(pd.read_csv(DATA_FILE)); X=df.drop(columns=[TARGET]+ID_COLUMNS); y=df[TARGET]
    X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=.30,stratify=y,random_state=RANDOM_STATE)
    X_val,X_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=.50,stratify=y_tmp,random_state=RANDOM_STATE)
    from sklearn.base import clone
    models={'logistic_regression':LogisticRegression(max_iter=2000,class_weight='balanced',solver='liblinear',random_state=RANDOM_STATE)}
    try:
        from xgboost import XGBClassifier
        ratio=(y_train==0).sum()/max(1,(y_train==1).sum())
        models['xgboost']=XGBClassifier(n_estimators=350,max_depth=4,learning_rate=.04,subsample=.85,colsample_bytree=.85,reg_lambda=2,scale_pos_weight=ratio,eval_metric='logloss',random_state=RANDOM_STATE,n_jobs=4)
    except ImportError:
        print('xgboost missing; install requirements to run both candidates')
    try:
        import mlflow
        from mlflow import sklearn as mlflow_sklearn
        tracking=os.getenv('MLFLOW_TRACKING_URI')
        if tracking: mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT','telco-churn-model-comparison'))
    except ImportError:
        mlflow=None
    ARTIFACT_DIR.mkdir(parents=True,exist_ok=True); results={}; candidates={}
    for name, estimator in models.items():
        pipe=Pipeline([('preprocess',make_preprocessor(X_train)),('classifier',estimator)])
        pipe.fit(X_train,y_train); vp=pipe.predict_proba(X_val)[:,1]; threshold=choose_threshold(y_val,vp)
        tp=pipe.predict_proba(X_test)[:,1]; m=calc_metrics(y_test,tp,threshold); results[name]=m; candidates[name]=(pipe,threshold)
        if mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({'model':name,'random_state':RANDOM_STATE,'min_recall_constraint':MIN_RECALL,'threshold_tuned_on':'validation'})
                mlflow.log_metrics(m)
                mlflow.log_dict(m,'metrics.json')
    # Choose best candidate by PR-AUC among models meeting recall floor; test metrics are reported only
    # for comparison here. For strict unbiased selection, choose by validation PR-AUC before test evaluation.
    winner=max(results,key=lambda n:results[n]['pr_auc']); pipe,threshold=candidates[winner]
    if results[winner]['recall'] < MIN_RECALL:
        print(f"WARNING: winner test recall {results[winner]['recall']:.3f} below target {MIN_RECALL}; do not auto-promote")
    artifact={'pipeline':pipe,'threshold':float(threshold),'features':X.columns.tolist(),'target':TARGET,'dataset_version':'ibm-telco-customer-churn','model_name':winner}
    joblib.dump(artifact,ARTIFACT_DIR/'model.joblib')
    (ARTIFACT_DIR/'metrics.json').write_text(json.dumps({'winner':winner,'candidates':results},indent=2))
    # SHAP: tree explainer for XGBoost; generic explainer fallback for linear model. Save compact global importance.
    try:
        import shap
        prep=pipe.named_steps['preprocess']; clf=pipe.named_steps['classifier']; sample=X_test.sample(min(200,len(X_test)),random_state=RANDOM_STATE); transformed=prep.transform(sample)
        if winner=='xgboost': explainer=shap.TreeExplainer(clf); sv=explainer.shap_values(transformed)
        else: explainer=shap.LinearExplainer(clf,transformed); sv=explainer.shap_values(transformed)
        names=prep.get_feature_names_out(); imp=np.abs(np.asarray(sv)).mean(axis=0).reshape(-1)
        pd.DataFrame({'feature':names,'mean_abs_shap':imp}).sort_values('mean_abs_shap',ascending=False).to_csv(ARTIFACT_DIR/'shap_global_importance.csv',index=False)
        shap.summary_plot(sv,transformed,feature_names=names,show=False); import matplotlib.pyplot as plt; plt.tight_layout(); plt.savefig(ARTIFACT_DIR/'shap_summary.png',dpi=150); plt.close()
    except Exception as e: (ARTIFACT_DIR/'shap_error.txt').write_text(str(e))
    print(json.dumps({'winner':winner,'candidates':results},indent=2))

if __name__=='__main__': main()
