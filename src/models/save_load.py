from __future__ import annotations

from pathlib import Path
import joblib


def save_pipeline(pipeline, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str | Path):
    path = Path(path)
    return joblib.load(path)


