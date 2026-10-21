"""Test doubles: a seeded synthetic OHLCV random walk and an in-memory provider."""

from datetime import date

import numpy as np
import pandas as pd

from cryptosignal.data.providers import DataProviderError

TODAY = date(2024, 1, 1)


def make_ohlcv(n: int = 900, *, seed: int = 7, end: date = TODAY) -> pd.DataFrame:
    """Daily bars ending the day before ``end`` (so nothing is dropped as incomplete)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(
        end=pd.Timestamp(end) - pd.Timedelta(days=1), periods=n, freq="D"
    ).as_unit("ns")
    log_returns = rng.normal(0.0005, 0.03, n)
    close = 20_000 * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.002, n))
    wick = np.abs(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    volume = rng.lognormal(mean=20, sigma=0.4, size=n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )
    df.index.name = "date"
    return df


class FakeProvider:
    name = "fake"

    def __init__(self, frames: dict[str, pd.DataFrame] | None = None) -> None:
        self.frames = frames if frames is not None else {}
        self.calls = 0
        self.fail = False

    def fetch(self, ticker: str, period: str = "max") -> pd.DataFrame:
        self.calls += 1
        if self.fail:
            raise DataProviderError("provider is down")
        if ticker not in self.frames:
            raise DataProviderError(f"unknown ticker {ticker}")
        return self.frames[ticker].copy()


def perturb_after(ohlcv: pd.DataFrame, position: int, seed: int = 0) -> pd.DataFrame:
    """Return a copy where every bar after ``position`` is replaced by random garbage."""
    rng = np.random.default_rng(seed)
    future = ohlcv.copy()
    n_future = len(future) - position - 1
    scale = rng.uniform(0.2, 5.0, size=(n_future, 1))
    future.iloc[position + 1 :] = future.iloc[position + 1 :].to_numpy() * scale
    return future
