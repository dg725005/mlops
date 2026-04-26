import pandas as pd
import joblib
import json
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_register():
    # 1. Load Data and Params
    df = pd.read_csv(r'E:\DESKTOP\mlops_iris\dataset\Iris.csv').iloc[75:,:]
    df = df.reset_index(drop=True)
    with open('params.json', 'r') as f:
        params = json.load(f)
    
    X, y = df.drop(columns= ['Id','Species'], axis=1), pd.get_dummies(df['Species']).replace({True:1.0,False:0.0})

    X.to_csv("X_data.csv")
    y.to_csv("y_data.csv")
    ######################################################################################################
    mlflow.set_experiment("Species_Spotting_Project")
    
    with mlflow.start_run() as run:
        # 2. Train
        model = RandomForestClassifier(**params)
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))

        import joblib
        import os

        # Create a folder for models if it doesn't exist
        #os.makedirs('models', exist_ok=True)

        # Save the file
        joblib.dump(model, 'models/model.pkl')
        
        # 3. Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "trained_model")
        
        # 4. AUTOMATIC DEPLOYMENT LOGIC
        client = MlflowClient()
        model_name = "Species_Spotting_Project"
        
        # Get current production accuracy
        try:
            prod_ver = client.get_latest_versions(model_name, stages=["Production"])[0]
            prod_acc = client.get_run(prod_ver.run_id).data.metrics['accuracy']
        except:
            prod_acc = 0  # No model in production yet

        # The "Gatekeeper" Check
        if acc > prod_acc:
            model_uri = f"runs:/{run.info.run_id}/trained_model"
            mv = mlflow.register_model(model_uri, model_name)
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Production", archive_existing_versions=True
            )
            print("New model promoted to Production!")

if __name__ == "__main__":
    train_and_register()