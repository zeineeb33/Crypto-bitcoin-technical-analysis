"""Technical indicators and model features, computed strictly without look-ahead.

Every value at date ``t`` depends only on bars ``<= t`` (rolling windows, ``pct_change``,
recursive EMAs). The *target* is the only column that looks forward: it is the direction
of the next close and is kept separate from the features.

Indicators reuse the original notebook (SMA, EMA 12/26, MACD + signal, 7-day volatility,
volume); model features are their scale-free versions so one model can be trained across
very different price regimes.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cryptosignal.config import FeatureSettings


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def compute_indicators(ohlcv: pd.DataFrame, cfg: FeatureSettings) -> pd.DataFrame:
    """Human-readable indicators (price units), used by the dashboard and the baselines."""
    close = ohlcv["close"]
    ema_fast = close.ewm(span=cfg.ema_fast, adjust=False).mean()
    ema_slow = close.ewm(span=cfg.ema_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=cfg.macd_signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "sma_fast": close.rolling(cfg.sma_fast).mean(),
            "sma_slow": close.rolling(cfg.sma_slow).mean(),
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd - macd_signal,
            "rsi": rsi(close, cfg.rsi_window),
            # notebook definition: rolling std of daily % change, in percent
            "volatility": close.pct_change().rolling(cfg.vol_fast).std() * 100,
        },
        index=ohlcv.index,
    )


def build_features(ohlcv: pd.DataFrame, cfg: FeatureSettings) -> pd.DataFrame:
    """Scale-free model features. Rows inside the warm-up period contain NaNs."""
    close, volume = ohlcv["close"], ohlcv["volume"]
    ind = compute_indicators(ohlcv, cfg)
    ret = close.pct_change()
    log_volume = np.log1p(volume)

    features: dict[str, pd.Series] = {"ret_1d": ret}
    for lag in range(1, cfg.return_lags):
        features[f"ret_1d_lag{lag}"] = ret.shift(lag)
    for window in cfg.momentum_windows:
        features[f"ret_{window}d"] = close.pct_change(window)

    vol_fast = ret.rolling(cfg.vol_fast).std()
    vol_slow = ret.rolling(cfg.vol_slow).std()
    features |= {
        "close_to_sma_fast": close / ind["sma_fast"] - 1,
        "close_to_sma_slow": close / ind["sma_slow"] - 1,
        "sma_fast_to_slow": ind["sma_fast"] / ind["sma_slow"] - 1,
        "macd_norm": ind["macd"] / close,
        "macd_signal_norm": ind["macd_signal"] / close,
        "macd_hist_norm": ind["macd_hist"] / close,
        "rsi": ind["rsi"] / 100,
        "vol_fast": vol_fast,
        "vol_slow": vol_slow,
        "vol_ratio": vol_fast / vol_slow,
        "volume_z": (log_volume - log_volume.rolling(cfg.volume_window).mean())
        / log_volume.rolling(cfg.volume_window).std(),
        "volume_change": log_volume.diff(),
        "hl_range": (ohlcv["high"] - ohlcv["low"]) / close,
        "body_return": close / ohlcv["open"] - 1,
    }
    out = pd.DataFrame(features, index=ohlcv.index)
    return out.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class Dataset:
    """Aligned features / target / forward return, one row per daily bar.

    ``target[t] = 1`` if ``close[t+1] > close[t]``; ``forward_return[t]`` is the return
    earned by holding from close ``t`` to close ``t+1``. Both are NaN on the last bar,
    which is the row used for the live prediction.
    """

    features: pd.DataFrame
    target: pd.Series
    forward_return: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    def labelled(self) -> "Dataset":
        mask = self.target.notna()
        return Dataset(
            self.features[mask], self.target[mask].astype(int), self.forward_return[mask]
        )

    def latest(self) -> pd.DataFrame:
        return self.features.iloc[[-1]]


def make_dataset(ohlcv: pd.DataFrame, cfg: FeatureSettings) -> Dataset:
    features = build_features(ohlcv, cfg)
    close = ohlcv["close"]
    next_close = close.shift(-1)
    forward_return = next_close / close - 1
    target = (next_close > close).astype(float).where(next_close.notna())

    complete = features.notna().all(axis=1)
    return Dataset(features[complete], target[complete], forward_return[complete])
