"""Feature engineering, including the look-ahead (leakage) guards."""

import numpy as np
import pandas as pd
import pytest

from cryptosignal.config import FeatureSettings
from cryptosignal.features import build_features, compute_indicators, make_dataset, rsi
from tests.fakes import perturb_after

CFG = FeatureSettings()


@pytest.mark.parametrize("position", [100, 450, 898])
def test_features_do_not_look_ahead(ohlcv: pd.DataFrame, position: int) -> None:
    """Features at t must be identical whatever happens after t."""
    original = build_features(ohlcv, CFG).iloc[: position + 1]
    perturbed = build_features(perturb_after(ohlcv, position), CFG).iloc[: position + 1]
    pd.testing.assert_frame_equal(original, perturbed)


@pytest.mark.parametrize("position", [100, 450])
def test_indicators_do_not_look_ahead(ohlcv: pd.DataFrame, position: int) -> None:
    original = compute_indicators(ohlcv, CFG).iloc[: position + 1]
    perturbed = compute_indicators(perturb_after(ohlcv, position), CFG).iloc[: position + 1]
    pd.testing.assert_frame_equal(original, perturbed)


def test_target_is_next_day_direction(ohlcv: pd.DataFrame) -> None:
    data = make_dataset(ohlcv, CFG)
    close = ohlcv["close"]
    for date in data.features.index[:-1][::97]:
        nxt = close.index[close.index.get_loc(date) + 1]
        assert data.target[date] == float(close[nxt] > close[date])
        assert data.forward_return[date] == pytest.approx(close[nxt] / close[date] - 1)


def test_latest_row_is_unlabelled_and_complete(ohlcv: pd.DataFrame) -> None:
    data = make_dataset(ohlcv, CFG)
    assert data.features.index[-1] == ohlcv.index[-1]
    assert np.isnan(data.target.iloc[-1])
    assert data.latest().notna().all(axis=None)
    labelled = data.labelled()
    assert len(labelled.target) == len(data.target) - 1
    assert set(labelled.target.unique()) <= {0, 1}


def test_features_are_finite_after_warmup(ohlcv: pd.DataFrame) -> None:
    features = make_dataset(ohlcv, CFG).features
    assert np.isfinite(features.to_numpy()).all()
    # warm-up is bounded by the longest window
    assert len(features) >= len(ohlcv) - max(CFG.sma_slow, CFG.vol_slow, *CFG.momentum_windows) - 1


def test_features_are_scale_free(ohlcv: pd.DataFrame) -> None:
    """Multiplying prices by a constant must not change the features."""
    scaled = ohlcv.copy()
    scaled[["open", "high", "low", "close"]] *= 1000
    pd.testing.assert_frame_equal(build_features(ohlcv, CFG), build_features(scaled, CFG))


def test_rsi_bounds_and_extremes() -> None:
    rising = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(rising).dropna().eq(100).all()
    noisy = pd.Series(np.random.default_rng(0).normal(0, 1, 500).cumsum() + 100)
    values = rsi(noisy).dropna()
    assert values.between(0, 100).all()


def test_macd_matches_notebook_definition(ohlcv: pd.DataFrame) -> None:
    ind = compute_indicators(ohlcv, CFG)
    ema12 = ohlcv["close"].ewm(span=12, adjust=False).mean()
    ema26 = ohlcv["close"].ewm(span=26, adjust=False).mean()
    pd.testing.assert_series_equal(ind["macd"], ema12 - ema26, check_names=False)


def test_leakage_guard_catches_a_leaky_feature(ohlcv: pd.DataFrame) -> None:
    """Sanity check of the guard itself: a centred window peeks at the future."""

    def leaky(frame: pd.DataFrame) -> pd.Series:
        return frame["close"].rolling(5, center=True).mean().iloc[:451]

    assert not leaky(ohlcv).equals(leaky(perturb_after(ohlcv, 450)))
