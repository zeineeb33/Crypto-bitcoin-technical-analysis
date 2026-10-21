"""Market data providers behind a small interface, so they can be mocked or swapped.

Every provider returns the canonical OHLCV frame: a tz-naive daily ``DatetimeIndex``
named ``date`` (UTC calendar days) and float columns ``open, high, low, close, volume``.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from cryptosignal.config import PERIOD_PATTERN

logger = logging.getLogger(__name__)

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class DataProviderError(RuntimeError):
    """The provider could not return data (network error, rate limit, unknown symbol...)."""


@runtime_checkable
class MarketDataProvider(Protocol):
    name: str

    def fetch(self, ticker: str, period: str = "max") -> pd.DataFrame:
        """Return daily OHLCV bars for ``ticker`` in the canonical format."""
        ...


def normalise_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a provider frame (any column case, tz-aware or naive index) to canonical form."""
    df = raw.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataProviderError(f"provider frame is missing columns: {missing}")
    df = df.loc[:, list(OHLCV_COLUMNS)].astype("float64")

    index = pd.DatetimeIndex(pd.to_datetime(df.index))
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    df.index = index.normalize().as_unit("ns")
    df.index.name = "date"
    return df.sort_index()


class YFinanceProvider:
    """Yahoo Finance via ``yfinance``, with retries and exponential backoff."""

    name = "yfinance"

    def __init__(
        self,
        retries: int = 3,
        backoff_seconds: float = 2.0,
        timeout_seconds: float = 20.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self._sleep = sleep

    def fetch(self, ticker: str, period: str = "max") -> pd.DataFrame:
        if not PERIOD_PATTERN.match(period):
            raise ValueError(f"invalid period: {period!r}")
        import yfinance as yf  # noqa: PLC0415 - heavy import, only needed for live fetches

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                raw = yf.Ticker(ticker).history(
                    period=period,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    timeout=self.timeout_seconds,
                )
                if raw is None or raw.empty:
                    raise DataProviderError(f"no data returned for {ticker!r}")
                df = normalise_ohlcv(raw)
                logger.info("fetched %d bars for %s from %s", len(df), ticker, self.name)
                return df
            except Exception as exc:  # yfinance raises many unrelated exception types
                last_error = exc
                logger.warning(
                    "fetch %s failed (attempt %d/%d): %s", ticker, attempt, self.retries, exc
                )
                if attempt < self.retries:
                    self._sleep(self.backoff_seconds * 2 ** (attempt - 1))
        raise DataProviderError(f"{self.name}: could not fetch {ticker!r}: {last_error}")


class CSVProvider:
    """Reads bars previously saved by ``cryptosignal ingest`` (offline, reproducible runs)."""

    name = "csv"

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def path_for(self, ticker: str) -> Path:
        return self.data_dir / "raw" / f"{ticker}.csv"

    def fetch(self, ticker: str, period: str = "max") -> pd.DataFrame:
        path = self.path_for(ticker)
        if not path.is_file():
            raise DataProviderError(f"{path} not found - run `cryptosignal ingest` first")
        raw = pd.read_csv(path, index_col="date", parse_dates=["date"])
        return normalise_ohlcv(raw)

    def save(self, ticker: str, df: pd.DataFrame) -> Path:
        path = self.path_for(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index_label="date", float_format="%.12g")
        return path
