"""Per-ticker cache: fetch -> validate -> train, refreshed lazily.

Designed for a small, sleeping container (Render free tier): nothing heavy happens at
import time, the first request (or the background warm-up) builds a snapshot, later
requests reuse it until ``refresh_minutes`` have passed. If the data provider fails, the
last good snapshot keeps being served (flagged ``stale``); with no snapshot at all the
API answers 503 and retries the provider only after a cool-down, so an outage never turns
into a crash loop or a request storm against Yahoo.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd

from cryptosignal.backtest import BacktestReport, run_backtest
from cryptosignal.config import Settings, normalise_ticker
from cryptosignal.data.ingest import load_market_data
from cryptosignal.data.providers import DataProviderError, MarketDataProvider
from cryptosignal.data.validation import DataValidationError
from cryptosignal.features import Dataset, compute_indicators, make_dataset
from cryptosignal.models import TrainedModel, train_model

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]


class UnknownTickerError(KeyError):
    pass


class ServiceUnavailableError(RuntimeError):
    def __init__(self, message: str, retry_after: int) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class Snapshot:
    ohlcv: pd.DataFrame
    dataset: Dataset
    indicators: pd.DataFrame
    model: TrainedModel
    refreshed_at: datetime
    source: str
    backtest: BacktestReport | None = None
    backtest_at: datetime | None = None


@dataclass
class _TickerState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    backtest_lock: threading.Lock = field(default_factory=threading.Lock)
    snapshot: Snapshot | None = None
    last_error: str | None = None
    last_failure: datetime | None = None


class SignalService:
    def __init__(
        self,
        settings: Settings,
        provider: MarketDataProvider,
        clock: Clock = lambda: datetime.now(UTC),
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.clock = clock
        self.started_at = clock()
        self._states = {ticker: _TickerState() for ticker in settings.tickers}

    @property
    def tickers(self) -> list[str]:
        return list(self._states)

    def _state(self, ticker: str) -> tuple[str, _TickerState]:
        try:
            key = normalise_ticker(ticker)
        except ValueError as exc:
            raise UnknownTickerError(ticker) from exc
        if key not in self._states:
            raise UnknownTickerError(ticker)
        return key, self._states[key]

    def is_stale(self, snapshot: Snapshot) -> bool:
        age = self.clock() - snapshot.refreshed_at
        return age >= timedelta(minutes=self.settings.serve.refresh_minutes)

    def status(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for ticker, state in self._states.items():
            snap = state.snapshot
            out[ticker] = {
                "ready": snap is not None,
                "stale": snap is not None and self.is_stale(snap),
                "data_end": snap.ohlcv.index[-1].date() if snap else None,
                "refreshed_at": snap.refreshed_at if snap else None,
                "last_error": state.last_error,
            }
        return out

    def snapshot(self, ticker: str) -> Snapshot:
        key, state = self._state(ticker)
        with state.lock:
            snap, now = state.snapshot, self.clock()
            if snap is not None and not self.is_stale(snap):
                return snap
            cooldown = timedelta(seconds=self.settings.serve.retry_after_seconds)
            if state.last_failure is not None and now - state.last_failure < cooldown:
                if snap is not None:
                    return snap
                remaining = cooldown - (now - state.last_failure)
                raise ServiceUnavailableError(
                    f"market data for {key} is unavailable: {state.last_error}",
                    retry_after=max(1, int(remaining.total_seconds())),
                )
            try:
                state.snapshot = self._build(key)
            except (DataProviderError, DataValidationError, ValueError) as exc:
                logger.error("refresh of %s failed: %s", key, exc)
                state.last_error, state.last_failure = str(exc), now
                if snap is not None:
                    logger.warning("serving stale %s snapshot from %s", key, snap.refreshed_at)
                    return snap
                raise ServiceUnavailableError(
                    f"market data for {key} is unavailable: {exc}",
                    retry_after=self.settings.serve.retry_after_seconds,
                ) from exc
            state.last_error = state.last_failure = None
            return state.snapshot

    def _build(self, ticker: str) -> Snapshot:
        s = self.settings
        ohlcv, _ = load_market_data(self.provider, ticker, s.data)
        dataset = make_dataset(ohlcv, s.features)
        model = train_model(dataset, ticker=ticker, kind=s.model.kind, cfg=s.model)
        return Snapshot(
            ohlcv=ohlcv,
            dataset=dataset,
            indicators=compute_indicators(ohlcv, s.features),
            model=model,
            refreshed_at=self.clock(),
            source=self.provider.name,
        )

    def backtest(self, ticker: str) -> Snapshot:
        """Snapshot with its walk-forward backtest computed (lazily, once per snapshot)."""
        _, state = self._state(ticker)
        snap = self.snapshot(ticker)
        with state.backtest_lock:
            if snap.backtest is None:
                s = self.settings
                snap.backtest = run_backtest(
                    snap.ohlcv,
                    ticker=snap.model.metadata.ticker,
                    models=s.serve.backtest_models,
                    features_cfg=s.features,
                    model_cfg=s.model,
                    cfg=s.backtest,
                )
                snap.backtest_at = self.clock()
        return snap

    def warmup(self) -> None:
        """Prepare every ticker (background thread at startup): signals first, then backtests."""
        for step in (self.snapshot, self.backtest):
            for ticker in self.tickers:
                try:
                    step(ticker)
                except Exception:  # never let warm-up kill the process
                    logger.exception("warm-up %s failed for %s", step.__name__, ticker)
        logger.info("warm-up finished")
