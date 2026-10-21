"""Model factory, training and persistence.

Two deliberately small models: a regularised logistic regression (linear baseline) and
scikit-learn's histogram gradient boosting (non-linear, no extra native dependency, fits
in a 512 MB container). Scaling lives *inside* the pipeline so it is re-fitted on each
training window and never sees test data.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cryptosignal.config import ModelKind, ModelSettings
from cryptosignal.features import Dataset

logger = logging.getLogger(__name__)

MODEL_LABELS: dict[str, str] = {
    "logreg": "Logistic regression",
    "hgb": "Gradient boosting (HistGB)",
}


def make_model(kind: ModelKind, cfg: ModelSettings) -> ClassifierMixin:
    if kind == "logreg":
        return cast(
            ClassifierMixin,
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=cfg.logreg_c, max_iter=2000)),
                ]
            ),
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=cfg.hgb_max_iter,
            learning_rate=cfg.hgb_learning_rate,
            max_depth=cfg.hgb_max_depth,
            min_samples_leaf=cfg.hgb_min_samples_leaf,
            l2_regularization=cfg.hgb_l2_regularization,
            early_stopping=False,  # deterministic and no hidden validation split
            random_state=cfg.random_state,
        )
    raise ValueError(f"unknown model kind: {kind!r}")


def predict_up_proba(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    """P(next close > close) for each row."""
    classes = list(estimator.classes_)
    return np.asarray(estimator.predict_proba(features)[:, classes.index(1)], dtype=float)


@dataclass(frozen=True)
class ModelMetadata:
    ticker: str
    kind: str
    version: str
    trained_at: str
    data_start: str
    data_end: str
    n_samples: int
    positive_rate: float
    feature_names: list[str]
    params: dict[str, Any]
    sklearn_version: str


@dataclass
class TrainedModel:
    estimator: Any
    metadata: ModelMetadata

    def predict_up_proba(self, features: pd.DataFrame) -> np.ndarray:
        return predict_up_proba(self.estimator, features[self.metadata.feature_names])

    def save(self, models_dir: Path) -> Path:
        directory = models_dir / self.metadata.ticker
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.metadata.kind}.joblib"
        joblib.dump(self.estimator, path)
        path.with_suffix(".json").write_text(json.dumps(asdict(self.metadata), indent=2))
        logger.info("saved %s to %s", self.metadata.version, path)
        return path

    @classmethod
    def load(cls, models_dir: Path, ticker: str, kind: str) -> "TrainedModel":
        """Load a model written by :meth:`save`. Only load files you produced (pickle)."""
        path = models_dir / ticker / f"{kind}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found - run `cryptosignal train` first")
        metadata = ModelMetadata(**json.loads(path.with_suffix(".json").read_text()))
        if metadata.sklearn_version != sklearn.__version__:
            logger.warning(
                "model trained with scikit-learn %s, running %s",
                metadata.sklearn_version,
                sklearn.__version__,
            )
        return cls(joblib.load(path), metadata)


def model_version(
    kind: str, data_end: pd.Timestamp, params: dict[str, Any], features: list[str]
) -> str:
    """Deterministic version: same data + config => same version string."""
    fingerprint = json.dumps({"params": params, "features": features}, sort_keys=True, default=str)
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:8]
    return f"{kind}-{data_end:%Y%m%d}-{digest}"


def train_model(
    dataset: Dataset, *, ticker: str, kind: ModelKind, cfg: ModelSettings
) -> TrainedModel:
    """Fit on every labelled row (i.e. all bars whose next close is known)."""
    data = dataset.labelled()
    if data.target.nunique() < 2:
        raise ValueError("training data contains a single class")
    estimator = make_model(kind, cfg)
    estimator.fit(data.features, data.target)

    params = {
        k: v for k, v in cfg.model_dump().items() if k.startswith(kind) or k == "random_state"
    }
    metadata = ModelMetadata(
        ticker=ticker,
        kind=kind,
        version=model_version(kind, data.features.index[-1], params, data.feature_names),
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        data_start=str(data.features.index[0].date()),
        data_end=str(data.features.index[-1].date()),
        n_samples=len(data.target),
        positive_rate=round(float(data.target.mean()), 4),
        feature_names=data.feature_names,
        params=params,
        sklearn_version=sklearn.__version__,
    )
    logger.info("%s: trained %s on %d samples", ticker, metadata.version, metadata.n_samples)
    return TrainedModel(estimator, metadata)
