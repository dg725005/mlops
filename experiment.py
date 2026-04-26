import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json

def objective(trial):
    df = pd.read_csv("E:\DESKTOP\mlops_iris\dataset\Iris.csv").iloc[:75,:]
    df = df.reset_index(drop=True)
    X, y = df.drop(columns = ['Id', 'Species'], axis=1), pd.get_dummies(df[['Species']], columns=['Species']).replace({True:1.0,False:0.0})

    # Define search space
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return cross_val_score(clf, X, y, cv=3).mean()

if __name__ == "__main__":

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    # Save best params to a file for the DVC pipeline to use
    with open('E:\DESKTOP\mlops_iris\params.json', 'w') as f:
        json.dump(study.best_params, f)