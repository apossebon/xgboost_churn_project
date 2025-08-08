from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    candidate_cols = [c for c in df.columns if c != target]
    numeric_cols: List[str] = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols: List[str] = [c for c in candidate_cols if not pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target: str,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols, categorical_cols = infer_feature_types(df, target)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


