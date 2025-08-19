from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


@dataclass
class TrainResult:
    pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray


def train_xgboost_classifier(
    df: pd.DataFrame,
    target: str,
    preprocessor,
    params: Dict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model = XGBClassifier(**params)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipe.predict(X_test)

    return TrainResult(
        pipeline=pipe, X_test=X_test, y_test=y_test, y_pred=y_pred, y_proba=y_proba
    )


