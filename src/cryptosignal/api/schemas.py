"""Request/response models (they also drive the OpenAPI docs at /docs)."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

DISCLAIMER = "Educational project - not financial advice."


class TickerStatus(BaseModel):
    ready: bool = Field(description="A model is trained and data is cached for this ticker.")
    stale: bool = Field(description="Cached data is older than the refresh interval.")
    data_end: date | None = Field(None, description="Date of the latest closed daily bar.")
    refreshed_at: datetime | None = None
    last_error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    uptime_seconds: float
    tickers: dict[str, TickerStatus]


class SignalResponse(BaseModel):
    ticker: str = Field(examples=["BTC-USD"])
    as_of: date = Field(description="Close of this daily bar is the latest input.")
    prediction_for: date = Field(description="Bar whose close direction is predicted.")
    probability_up: float = Field(ge=0, le=1, description="P(next close > latest close).")
    direction: Literal["up", "down"]
    signal: Literal["long", "flat"] = Field(description="Long when probability_up > threshold.")
    threshold: float
    model_kind: str
    model_version: str
    trained_at: datetime
    training_samples: int
    data_source: str
    refreshed_at: datetime
    stale: bool
    disclaimer: str = DISCLAIMER


class IndicatorPoint(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    sma_fast: float | None
    sma_slow: float | None
    ema_fast: float | None
    ema_slow: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    rsi: float | None
    volatility: float | None = Field(description="Rolling std of daily % change, in percent.")


class IndicatorsResponse(BaseModel):
    ticker: str
    windows: dict[str, int] = Field(description="Indicator window lengths, in days.")
    points: list[IndicatorPoint]


class ClassificationOut(BaseModel):
    accuracy: float
    accuracy_ci95: tuple[float, float]
    roc_auc: float | None
    up_rate_predicted: float
    n: int


class StrategyOut(BaseModel):
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    exposure: float
    trades: int


class EquityCurves(BaseModel):
    dates: list[date]
    series: dict[str, list[float]] = Field(description="Growth of 1 unit, net of costs.")


class BacktestResponse(BaseModel):
    ticker: str
    test_start: date
    test_end: date
    n_test: int
    up_rate: float = Field(description="Share of up days in the test period.")
    cost_bps: float
    retrain_every: int
    min_train_days: int
    models: list[str]
    labels: dict[str, str]
    classification: dict[str, ClassificationOut]
    strategies: dict[str, StrategyOut]
    equity: EquityCurves
    computed_at: datetime
    disclaimer: str = DISCLAIMER


class ErrorResponse(BaseModel):
    detail: str
