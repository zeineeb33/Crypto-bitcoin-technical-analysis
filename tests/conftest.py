"""Shared fixtures. Tests never touch the network (pytest-socket blocks it): market data
comes from a seeded synthetic random walk served by ``FakeProvider``."""

import os
from pathlib import Path

import pandas as pd
import pytest
import urllib3  # noqa: F401 - probes IPv6 with a socket at import; do it before sockets are blocked

from cryptosignal.config import Settings
from tests.fakes import FakeProvider, make_ohlcv


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider({"BTC-USD": make_ohlcv(seed=1), "ETH-USD": make_ohlcv(seed=2)})


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Small, fast configuration writing into a temporary directory."""
    return Settings(
        tickers=["BTC-USD", "ETH-USD"],
        artifacts_dir=tmp_path / "artifacts",
        models_dir=tmp_path / "models",
        data={"data_dir": tmp_path / "data", "min_rows": 300},  # type: ignore[arg-type]
        backtest={"min_train_days": 300, "retrain_every": 60},  # type: ignore[arg-type]
        model={"hgb_max_iter": 30},  # type: ignore[arg-type]
        serve={"warmup": False},  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's CRYPTOSIGNAL_* variables out of the tests."""
    for key in list(os.environ):
        if key.startswith("CRYPTOSIGNAL_"):
            monkeypatch.delenv(key)
