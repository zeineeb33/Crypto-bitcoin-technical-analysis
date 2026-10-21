from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptosignal.config import FeatureSettings, ModelSettings
from cryptosignal.features import make_dataset
from cryptosignal.models import TrainedModel, make_model, train_model

CFG = ModelSettings(hgb_max_iter=20)


@pytest.mark.parametrize("kind", ["logreg", "hgb"])
def test_train_save_load_round_trip(tmp_path: Path, ohlcv: pd.DataFrame, kind: str) -> None:
    dataset = make_dataset(ohlcv, FeatureSettings())
    model = train_model(dataset, ticker="BTC-USD", kind=kind, cfg=CFG)  # type: ignore[arg-type]

    proba = model.predict_up_proba(dataset.latest())
    assert proba.shape == (1,)
    assert 0 <= proba[0] <= 1
    assert model.metadata.n_samples == len(dataset.target) - 1
    assert model.metadata.version.startswith(f"{kind}-20231230-")

    model.save(tmp_path)
    loaded = TrainedModel.load(tmp_path, "BTC-USD", kind)
    assert loaded.metadata == model.metadata
    np.testing.assert_allclose(loaded.predict_up_proba(dataset.latest()), proba)


def test_version_is_deterministic(ohlcv: pd.DataFrame) -> None:
    dataset = make_dataset(ohlcv, FeatureSettings())
    a = train_model(dataset, ticker="X", kind="logreg", cfg=CFG).metadata.version
    b = train_model(dataset, ticker="X", kind="logreg", cfg=CFG).metadata.version
    c = train_model(dataset, ticker="X", kind="logreg", cfg=ModelSettings(logreg_c=1.0))
    assert a == b
    assert c.metadata.version != a


def test_load_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cryptosignal train"):
        TrainedModel.load(tmp_path, "BTC-USD", "logreg")


def test_unknown_model_kind() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        make_model("svm", CFG)  # type: ignore[arg-type]


def test_single_class_training_data_rejected(ohlcv: pd.DataFrame) -> None:
    rising = ohlcv.assign(close=np.linspace(100, 200, len(ohlcv)))
    with pytest.raises(ValueError, match="single class"):
        train_model(make_dataset(rising, FeatureSettings()), ticker="X", kind="logreg", cfg=CFG)
