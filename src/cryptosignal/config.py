"""Typed configuration.

Precedence (highest first): explicit kwargs > environment variables > YAML file > defaults.

* Environment variables use the ``CRYPTOSIGNAL_`` prefix and ``__`` for nesting,
  e.g. ``CRYPTOSIGNAL_BACKTEST__COST_BPS=5`` or ``CRYPTOSIGNAL_TICKERS='["BTC-USD"]'``.
* A YAML file can be passed with ``--config`` on the CLI or ``CRYPTOSIGNAL_CONFIG``.
"""

import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

ModelKind = Literal["logreg", "hgb"]
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-=^]{0,19}$")
PERIOD_PATTERN = re.compile(r"^(\d+(d|mo|y)|ytd|max)$")
CONFIG_ENV_VAR = "CRYPTOSIGNAL_CONFIG"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataSettings(_Section):
    period: str = Field("max", description="yfinance history period, e.g. '2y', '5y', 'max'.")
    data_dir: Path = Path("data")
    min_rows: int = Field(400, ge=50, description="Minimum number of daily bars required.")
    max_gap_days: int = Field(3, ge=1, description="Largest tolerated hole in the daily calendar.")
    drop_incomplete_bar: bool = Field(
        True, description="Drop today's (still forming) UTC daily bar before computing features."
    )

    @field_validator("period")
    @classmethod
    def _check_period(cls, period: str) -> str:
        if not PERIOD_PATTERN.match(period):
            raise ValueError(f"invalid period {period!r} (expected e.g. '730d', '5y', 'max')")
        return period


class FeatureSettings(_Section):
    sma_fast: int = 7
    sma_slow: int = 30
    ema_fast: int = 12
    ema_slow: int = 26
    macd_signal: int = 9
    rsi_window: int = 14
    vol_fast: int = 7
    vol_slow: int = 30
    volume_window: int = 20
    return_lags: int = Field(5, ge=1)
    momentum_windows: tuple[int, ...] = (7, 30)


class ModelSettings(_Section):
    kind: ModelKind = Field("logreg", description="Model used by `train`, `predict` and the API.")
    random_state: int = 42
    logreg_c: float = Field(0.1, gt=0)
    hgb_max_iter: int = 200
    hgb_learning_rate: float = 0.05
    hgb_max_depth: int = 3
    hgb_min_samples_leaf: int = 50
    hgb_l2_regularization: float = 1.0


class BacktestSettings(_Section):
    models: tuple[ModelKind, ...] = ("logreg", "hgb")
    min_train_days: int = Field(730, ge=100, description="Size of the first training window.")
    retrain_every: int = Field(30, ge=1, description="Refit the model every N test days.")
    window: Literal["expanding", "rolling"] = "expanding"
    rolling_window_days: int = Field(1095, ge=100)
    cost_bps: float = Field(10.0, ge=0, description="Cost per unit of turnover, in basis points.")
    threshold: float = Field(0.5, gt=0, lt=1, description="Go long when P(up) > threshold.")
    sma_fast: int = 50
    sma_slow: int = 200
    periods_per_year: int = 365  # crypto trades every day

    @model_validator(mode="after")
    def _check_sma(self) -> "BacktestSettings":
        if self.sma_fast >= self.sma_slow:
            raise ValueError("sma_fast must be smaller than sma_slow")
        return self


class ServeSettings(_Section):
    refresh_minutes: int = Field(360, ge=1, description="Refetch data and retrain after N minutes.")
    retry_after_seconds: int = Field(
        120, ge=1, description="Cool-down before retrying a failed data provider."
    )
    warmup: bool = Field(True, description="Prepare every ticker in the background at startup.")
    backtest_models: tuple[ModelKind, ...] = ("logreg",)
    indicator_days: int = Field(365, ge=30, le=5000)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CRYPTOSIGNAL_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    tickers: list[str] = Field(default_factory=lambda: ["BTC-USD", "ETH-USD"], min_length=1)
    log_level: str = "INFO"
    threads: int = Field(
        1, ge=1, description="BLAS/OpenMP threads; small daily datasets are fastest on one."
    )
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("models")
    data: DataSettings = Field(default_factory=DataSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    serve: ServeSettings = Field(default_factory=ServeSettings)

    @field_validator("tickers")
    @classmethod
    def _normalise_tickers(cls, tickers: list[str]) -> list[str]:
        normalised = [normalise_ticker(t) for t in tickers]
        if len(set(normalised)) != len(normalised):
            raise ValueError("duplicate tickers")
        return normalised

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, YamlConfigSettingsSource(settings_cls)


def normalise_ticker(ticker: str) -> str:
    value = ticker.strip().upper()
    if not TICKER_PATTERN.match(value):
        raise ValueError(f"invalid ticker symbol: {ticker!r}")
    return value


def load_settings(config_file: Path | None = None) -> Settings:
    """Load settings, optionally layering a YAML file under the environment variables."""
    if config_file is None and (env_path := os.environ.get(CONFIG_ENV_VAR)):
        config_file = Path(env_path)
    if config_file is None:
        return Settings()
    if not config_file.is_file():
        raise FileNotFoundError(f"config file not found: {config_file}")

    class _FileSettings(Settings):
        model_config = SettingsConfigDict(yaml_file=config_file)

    return _FileSettings()
