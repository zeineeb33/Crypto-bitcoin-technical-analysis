"""FastAPI application factory.

Run locally with ``cryptosignal serve`` (or ``uvicorn --factory cryptosignal.api.app:create_app``).
"""

import logging
import math
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Query, Request
from fastapi import Path as PathParam
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cryptosignal import __version__
from cryptosignal.api.schemas import (
    BacktestResponse,
    EquityCurves,
    ErrorResponse,
    HealthResponse,
    IndicatorPoint,
    IndicatorsResponse,
    SignalResponse,
    TickerStatus,
)
from cryptosignal.api.service import ServiceUnavailableError, SignalService, UnknownTickerError
from cryptosignal.config import Settings, load_settings
from cryptosignal.data.providers import MarketDataProvider, YFinanceProvider
from cryptosignal.runtime import configure_runtime

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
TickerParam = Annotated[str, PathParam(description="Configured ticker, e.g. BTC-USD.")]
ERRORS: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Ticker not configured"},
    503: {"model": ErrorResponse, "description": "Market data provider unavailable"},
}


def _clean(value: float | None) -> float | None:
    return None if value is None or math.isnan(value) else float(value)


def create_app(
    settings: Settings | None = None, provider: MarketDataProvider | None = None
) -> FastAPI:
    settings = settings or load_settings()
    configure_runtime(settings)
    service = SignalService(settings, provider or YFinanceProvider())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if settings.serve.warmup:
            threading.Thread(target=service.warmup, name="warmup", daemon=True).start()
        yield

    app = FastAPI(
        title="cryptosignal",
        version=__version__,
        summary="Next-day direction signals, technical indicators and walk-forward backtests.",
        description="Educational project - **not financial advice**.",
        lifespan=lifespan,
    )
    app.state.service = service
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.exception_handler(UnknownTickerError)
    async def _unknown(_: Request, exc: UnknownTickerError) -> JSONResponse:
        detail = f"unknown ticker {exc.args[0]!r}; configured: {', '.join(service.tickers)}"
        return JSONResponse(status_code=404, content={"detail": detail})

    @app.exception_handler(ServiceUnavailableError)
    async def _unavailable(_: Request, exc: ServiceUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc)},
            headers={"Retry-After": str(exc.retry_after)},
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Liveness + per-ticker cache status. Never calls the data provider."""
        return HealthResponse(
            version=__version__,
            uptime_seconds=round((service.clock() - service.started_at).total_seconds(), 1),
            tickers={t: TickerStatus.model_validate(s) for t, s in service.status().items()},
        )

    @app.get("/tickers", response_model=list[str], tags=["market"])
    def tickers() -> list[str]:
        return service.tickers

    @app.get("/signal/{ticker}", response_model=SignalResponse, responses=ERRORS, tags=["model"])
    def signal(ticker: TickerParam) -> SignalResponse:
        """Latest prediction for the next daily close, from a model trained on all history."""
        snap = service.snapshot(ticker)
        meta = snap.model.metadata
        latest = snap.dataset.latest()
        proba = float(snap.model.predict_up_proba(latest)[0])
        as_of = latest.index[-1].date()
        threshold = settings.backtest.threshold
        return SignalResponse(
            ticker=meta.ticker,
            as_of=as_of,
            prediction_for=as_of + timedelta(days=1),
            probability_up=round(proba, 4),
            direction="up" if proba > 0.5 else "down",
            signal="long" if proba > threshold else "flat",
            threshold=threshold,
            model_kind=meta.kind,
            model_version=meta.version,
            trained_at=datetime.fromisoformat(meta.trained_at),
            training_samples=meta.n_samples,
            data_source=snap.source,
            refreshed_at=snap.refreshed_at,
            stale=service.is_stale(snap),
        )

    @app.get(
        "/indicators/{ticker}", response_model=IndicatorsResponse, responses=ERRORS, tags=["market"]
    )
    def indicators(
        ticker: TickerParam,
        days: Annotated[
            int | None, Query(ge=30, le=5000, description="Most recent N bars.")
        ] = None,
    ) -> IndicatorsResponse:
        """Daily OHLCV with the notebook's indicators (SMA, EMA, MACD, RSI, volatility)."""
        snap = service.snapshot(ticker)
        n = days or settings.serve.indicator_days
        frame = snap.ohlcv.join(snap.indicators).tail(n)
        points = [
            IndicatorPoint.model_validate(
                {"date": ts.date(), **{k: _clean(v) for k, v in row.items()}}
            )
            for ts, row in zip(frame.index, frame.to_dict("records"), strict=True)
        ]
        f = settings.features
        return IndicatorsResponse(
            ticker=snap.model.metadata.ticker,
            windows={
                "sma_fast": f.sma_fast,
                "sma_slow": f.sma_slow,
                "ema_fast": f.ema_fast,
                "ema_slow": f.ema_slow,
                "macd_signal": f.macd_signal,
                "rsi": f.rsi_window,
                "volatility": f.vol_fast,
            },
            points=points,
        )

    @app.get(
        "/backtest/{ticker}", response_model=BacktestResponse, responses=ERRORS, tags=["model"]
    )
    def backtest(ticker: TickerParam) -> BacktestResponse:
        """Walk-forward backtest of the served model vs baselines (computed once per refresh)."""
        snap = service.backtest(ticker)
        report = snap.backtest
        assert report is not None and snap.backtest_at is not None
        payload = report.to_dict(include_equity=False)
        cfg = report.settings
        return BacktestResponse(
            **{k: payload[k] for k in ("ticker", "test_start", "test_end", "n_test", "up_rate")},
            cost_bps=cfg["cost_bps"],
            retrain_every=cfg["retrain_every"],
            min_train_days=cfg["min_train_days"],
            models=report.models,
            labels=report.labels,
            classification=payload["classification"],
            strategies=payload["strategies"],
            equity=EquityCurves(
                dates=[d.date() for d in report.equity.index],
                series={c: report.equity[c].round(5).tolist() for c in report.equity.columns},
            ),
            computed_at=snap.backtest_at,
        )

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
