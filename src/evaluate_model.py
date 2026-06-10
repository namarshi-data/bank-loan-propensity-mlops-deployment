"""Reusable model evaluation utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _get_positive_class_scores(model, X_test):
    """Return positive class scores for ROC-AUC when available."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    raise AttributeError(
        "Model must support predict_proba or decision_function for ROC-AUC."
    )


def evaluate_classifier(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name: str,
    print_report: bool = True,
) -> dict:
    """Evaluate a binary classifier and return performance metrics."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_score = _get_positive_class_scores(model, X_test)

    cm = confusion_matrix(y_test, y_test_pred)

    if cm.shape != (2, 2):
        raise ValueError("This evaluator expects a binary classification problem.")

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Model": model_name,
        "Train Accuracy": round(accuracy_score(y_train, y_train_pred), 4),
        "Test Accuracy": round(accuracy_score(y_test, y_test_pred), 4),
        "Precision": round(precision_score(y_test, y_test_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_test_pred, zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, y_test_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_test_score), 4),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
    }

    if print_report:
        print(model_name)
        print("\nConfusion Matrix")
        print(cm)
        print("\nClassification Report")
        print(classification_report(y_test, y_test_pred, zero_division=0))
        print("\nError Analysis")
        print(f"False Positives : {fp}")
        print(f"False Negatives : {fn}")

    return metrics


def metrics_to_dataframe(metrics: list[dict]) -> pd.DataFrame:
    """Convert a list of metric dictionaries into a DataFrame."""
    return pd.DataFrame(metrics)


def create_confusion_matrix_table(
    y_true,
    y_pred,
) -> pd.DataFrame:
    """Create a readable confusion matrix table."""
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=["Actual No Loan", "Actual Loan"],
        columns=["Predicted No", "Predicted Yes"],
    )
