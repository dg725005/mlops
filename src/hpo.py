import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json

def objective(trial):
    df = pd.read_csv('data/cleaned_data.csv')
    X, y = df.drop('Target', axis=1), df['Target']
    
    # Define search space
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return cross_val_score(clf, X, y, n_cv=3).mean()

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    # Save best params to a file for the DVC pipeline to use
    with open('params.json', 'w') as f:
        json.dump(study.best_params, f)