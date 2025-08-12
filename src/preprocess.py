# src/preprocess.py
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

NUMERIC_FEATURES = ['age', 'length_of_stay', 'num_prev_adm', 'comorbidity_score', 'labs_mean', 'med_changes']
CATEGORICAL_FEATURES = ['sex', 'discharge_disposition', 'insurance']

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder='drop')
    return preprocessor

def fit_transform(preprocessor, df):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['readmit_30d'].values
    X_trans = preprocessor.fit_transform(X)
    feature_names = get_feature_names(preprocessor)
    return X_trans, y, feature_names

def transform(preprocessor, df):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['readmit_30d'].values if 'readmit_30d' in df.columns else None
    X_trans = preprocessor.transform(X)
    return X_trans, y

def get_feature_names(preprocessor):
    # numeric names
    num_names = NUMERIC_FEATURES
    # categorical names from onehot encoder
    cat_transformer = None
    for name, trans, cols in preprocessor.transformers_:
        if name == 'cat':
            cat_transformer = trans
            break
    onehot: OneHotEncoder = cat_transformer.named_steps['onehot']
    cat_feature_names = onehot.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return num_names + cat_feature_names
