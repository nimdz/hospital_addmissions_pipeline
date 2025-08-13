# src/train.py
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from preprocess import build_preprocessor, fit_transform, get_feature_names
from generate_data import save_data

MODEL_PATH = 'outputs/model.joblib'
PREPROCESSOR_PATH = 'outputs/preprocessor.joblib'

def train_pipeline(train_csv='data/train.csv', save_paths=True):
    os.makedirs('outputs', exist_ok=True)

    if not os.path.exists(train_csv):
        print("Train csv not found, generating synthetic data...")
        save_data()

    df = pd.read_csv(train_csv)
    preprocessor = build_preprocessor()
    X, y, feature_names = fit_transform(preprocessor, df)

    # lightweight hyperparameter search
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=4,
        random_state=42
    )

    param_grid = {
        'n_estimators': [100],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    clf = GridSearchCV(xgb, param_grid, cv=cv, scoring='roc_auc', verbose=1, n_jobs=2)
    clf.fit(X, y)
    best = clf.best_estimator_
    print("Best params:", clf.best_params_)

    # one metric on training set (for quick inspection)
    train_pred = best.predict_proba(X)[:, 1]
    print("Train ROC AUC:", roc_auc_score(y, train_pred))

    if save_paths:
        joblib.dump(best, MODEL_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        # save feature names for mapping SHAP later
        joblib.dump(feature_names, 'outputs/feature_names.joblib')
        print(f"Saved model to {MODEL_PATH}, preprocessor to {PREPROCESSOR_PATH}")

    return best, preprocessor, feature_names

if __name__ == "__main__":
    train_pipeline()
