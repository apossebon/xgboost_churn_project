from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray


def evaluate_classification(y_true, y_pred, y_proba) -> EvalResult:
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred, zero_division=0)
    rec = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = metrics.roc_auc_score(y_true, y_proba)
    except Exception:
        roc = float("nan")
    cm = metrics.confusion_matrix(y_true, y_pred)
    return EvalResult(
        metrics={"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc},
        confusion_matrix=cm,
    )


def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
    auc_val = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Falso Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_confusion(cm: np.ndarray):
    plt.figure(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(cm).plot(values_format="d")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.show()


