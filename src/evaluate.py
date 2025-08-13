# src/evaluate.py
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix, classification_report

MODEL_PATH = 'outputs/model.joblib'
PREPROCESSOR_PATH = 'outputs/preprocessor.joblib'
FEATURE_NAMES = 'outputs/feature_names.joblib'

def evaluate(test_csv='data/test.csv'):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found, run train.py first")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES)

    df = pd.read_csv(test_csv)
    X_test, y_test = preprocessor.transform(df[preprocessor.feature_names_in_]), df['readmit_30d'].values  # alternative
    # safer transform via helper
    from preprocess import transform
    X_test, y_test = transform(preprocessor, df)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    print("ROC AUC:", roc_auc)
    print("Average Precision (PR AUC):", avg_prec)
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/roc_curve.png', dpi=150)
    plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, label=f'AP = {avg_prec:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('outputs/pr_curve.png', dpi=150)
    plt.close()

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.close()
    print("Saved plots to outputs/")

if __name__ == "__main__":
    evaluate()
