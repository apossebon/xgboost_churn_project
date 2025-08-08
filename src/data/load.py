from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def load_raw_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    return pd.read_csv(csv_path)


def generate_synthetic_churn(
    n_samples: int = 5000,
    n_numeric_features: int = 8,
    n_categorical_features: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_numeric_features,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],
        flip_y=0.01,
        random_state=random_state,
    )

    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(n_numeric_features)])
    rng = np.random.default_rng(random_state)
    for j in range(n_categorical_features):
        df[f"cat_{j}"] = rng.choice(["A", "B", "C"], size=n_samples, p=[0.5, 0.3, 0.2])

    df["churn"] = y.astype(int)
    return df


