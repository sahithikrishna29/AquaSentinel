
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred):
    """
    Calculate classification evaluation metrics.

    y_true: Actual class labels
    y_pred: Predicted class labels

    Returns evaluation metrics.
    """

    if len(y_true) == 0:
        raise ValueError("Actual labels cannot be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Actual and predicted labels must have equal lengths."
        )

    valid_classes = {0, 1, 2, 3}

    actual_classes = set(y_true)
    predicted_classes = set(y_pred)

    if not actual_classes.issubset(valid_classes):
        raise ValueError("Invalid actual class labels.")

    if not predicted_classes.issubset(valid_classes):
        raise ValueError("Invalid predicted class labels.")

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred,
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true, y_pred,
        labels=[0, 1, 2, 3]
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=[
            "Excellent",
            "Good",
            "Poor",
            "Dangerous"
        ],
        zero_division=0
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report
    }
