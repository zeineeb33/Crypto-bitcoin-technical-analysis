"""Ingestion step: fetch -> clean -> validate -> persist."""

import json
import logging
from datetime import date

import pandas as pd

from cryptosignal.config import DataSettings
from cryptosignal.data.providers import CSVProvider, MarketDataProvider
from cryptosignal.data.validation import ValidationReport, clean_ohlcv, validate_ohlcv

logger = logging.getLogger(__name__)


def load_market_data(
    provider: MarketDataProvider,
    ticker: str,
    settings: DataSettings,
    *,
    period: str | None = None,
    today: date | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Fetch bars and pass them through the quality gate (raises ``DataValidationError``)."""
    raw = provider.fetch(ticker, period or settings.period)
    df, cleaning_issues = clean_ohlcv(
        raw, drop_incomplete_bar=settings.drop_incomplete_bar, today=today
    )
    report = validate_ohlcv(
        df,
        ticker=ticker,
        min_rows=settings.min_rows,
        max_gap_days=settings.max_gap_days,
        today=today,
    )
    report.issues[:0] = cleaning_issues
    report.raise_for_errors()
    return df, report


def ingest(
    tickers: list[str],
    provider: MarketDataProvider,
    settings: DataSettings,
    *,
    today: date | None = None,
) -> dict[str, ValidationReport]:
    """Download and store validated bars as ``<data_dir>/raw/<TICKER>.csv``."""
    store = CSVProvider(settings.data_dir)
    reports: dict[str, ValidationReport] = {}
    for ticker in tickers:
        df, report = load_market_data(provider, ticker, settings, today=today)
        path = store.save(ticker, df)
        path.with_suffix(".validation.json").write_text(json.dumps(report.to_dict(), indent=2))
        logger.info(
            "%s: %d bars %s -> %s saved to %s", ticker, len(df), report.start, report.end, path
        )
        reports[ticker] = report
    return reports
