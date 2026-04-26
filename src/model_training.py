import pandas as pd
import joblib
import json
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_register():
    # 1. Load Data and Params
    df = pd.read_csv('data/cleaned_data.csv')
    with open('params.json', 'r') as f:
        params = json.load(f)
    
    X, y = df.drop('Target', axis=1), df['Target']
    
    mlflow.set_experiment("Student_Dropout_Project")
    
    with mlflow.start_run() as run:
        # 2. Train
        model = RandomForestClassifier(**params)
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        
        # 3. Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "risk_model")
        
        # 4. AUTOMATIC DEPLOYMENT LOGIC
        client = MlflowClient()
        model_name = "StudentDropoutModel"
        
        # Get current production accuracy
        try:
            prod_ver = client.get_latest_versions(model_name, stages=["Production"])[0]
            prod_acc = client.get_run(prod_ver.run_id).data.metrics['accuracy']
        except:
            prod_acc = 0  # No model in production yet

        # The "Gatekeeper" Check
        if acc > prod_acc:
            model_uri = f"runs:/{run.info.run_id}/risk_model"
            mv = mlflow.register_model(model_uri, model_name)
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Production", archive_existing_versions=True
            )
            print("New model promoted to Production!")

if __name__ == "__main__":
    train_and_register()