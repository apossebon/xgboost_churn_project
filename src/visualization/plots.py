from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def plot_roc(y_true, y_proba):
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


def plot_confusion_matrix(cm: np.ndarray):
    metrics.ConfusionMatrixDisplay(cm).plot(values_format="d")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.show()


