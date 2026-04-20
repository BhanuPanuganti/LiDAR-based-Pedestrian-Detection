import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    average_precision_score, precision_recall_curve
)

import os

VIS_DIR = os.path.join("outputs", "visuals")
os.makedirs(VIS_DIR, exist_ok=True)

def evaluate_model(model, X_test, y_test, threshold=0.9):

    # Probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    ap = average_precision_score(y_test, y_prob)

    print(f"\n--- Evaluation @ threshold={threshold} ---")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"AP       : {ap:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_prob, y_pred

def plot_pr_curve(y_test, y_prob):
    prec, rec, thresh = precision_recall_curve(y_test, y_prob)

    plt.figure()
    plt.plot(thresh, prec[:-1], label="Precision")
    plt.plot(thresh, rec[:-1], label="Recall")

    plt.xlabel("Threshold")
    plt.title("Precision vs Recall")
    plt.legend()

    save_path = os.path.join(VIS_DIR, "pr_curve.png")
    plt.savefig(save_path)
    print(f"[INFO] Saved: {save_path}")

    plt.close()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['BG','Ped'],
                yticklabels=['BG','Ped'])

    plt.title("Confusion Matrix")

    save_path = os.path.join(VIS_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"[INFO] Saved: {save_path}")

    plt.close()

def plot_feature_importance(model, feature_names):
    import pandas as pd

    plt.figure()
    pd.Series(model.feature_importances_, index=feature_names) \
        .sort_values() \
        .plot(kind='barh')

    plt.title("Feature Importance")

    save_path = os.path.join(VIS_DIR, "feature_importance.png")
    plt.savefig(save_path)
    print(f"[INFO] Saved: {save_path}")

    plt.close()