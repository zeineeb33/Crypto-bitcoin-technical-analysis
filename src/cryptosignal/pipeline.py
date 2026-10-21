"""Batch pipeline steps used by the CLI and the scheduled GitHub workflow.

ingest (data/raw/*.csv) -> train (models/<ticker>/<kind>.joblib) -> backtest
(artifacts/report.{json,md}, artifacts/model_card.md)
"""

import json
import logging
from pathlib import Path
from typing import Any

from cryptosignal.backtest import BacktestReport, run_backtest
from cryptosignal.config import ModelKind, Settings
from cryptosignal.data.ingest import load_market_data
from cryptosignal.data.providers import MarketDataProvider
from cryptosignal.features import make_dataset
from cryptosignal.models import ModelMetadata, TrainedModel, train_model
from cryptosignal.report import render_model_card, render_report

logger = logging.getLogger(__name__)


def train(
    tickers: list[str],
    settings: Settings,
    provider: MarketDataProvider,
    kind: ModelKind | None = None,
) -> dict[str, TrainedModel]:
    kind = kind or settings.model.kind
    models: dict[str, TrainedModel] = {}
    for ticker in tickers:
        ohlcv, _ = load_market_data(provider, ticker, settings.data)
        dataset = make_dataset(ohlcv, settings.features)
        model = train_model(dataset, ticker=ticker, kind=kind, cfg=settings.model)
        model.save(settings.models_dir)
        models[ticker] = model
    return models


def backtest(
    tickers: list[str], settings: Settings, provider: MarketDataProvider
) -> dict[str, BacktestReport]:
    """Run the walk-forward backtest and write report.json, report.md and model_card.md."""
    reports: dict[str, BacktestReport] = {}
    validation: dict[str, Any] = {}
    for ticker in tickers:
        ohlcv, quality = load_market_data(provider, ticker, settings.data)
        validation[ticker] = quality.to_dict()
        reports[ticker] = run_backtest(
            ohlcv,
            ticker=ticker,
            models=settings.backtest.models,
            features_cfg=settings.features,
            model_cfg=settings.model,
            cfg=settings.backtest,
        )
    write_reports(list(reports.values()), validation, settings)
    return reports


def _load_metadata(settings: Settings, tickers: list[str]) -> dict[str, ModelMetadata]:
    metadata: dict[str, ModelMetadata] = {}
    for ticker in tickers:
        path = settings.models_dir / ticker / f"{settings.model.kind}.json"
        if path.is_file():
            metadata[ticker] = ModelMetadata(**json.loads(path.read_text()))
    return metadata


def write_reports(
    reports: list[BacktestReport], validation: dict[str, Any], settings: Settings
) -> list[Path]:
    out = settings.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": settings.model_dump(mode="json"),
        "validation": validation,
        "reports": {r.ticker: r.to_dict() for r in reports},
    }
    paths = [out / "report.json", out / "report.md", out / "model_card.md"]
    paths[0].write_text(json.dumps(payload, indent=2))
    paths[1].write_text(render_report(reports))
    metadata = _load_metadata(settings, [r.ticker for r in reports])
    paths[2].write_text(render_model_card(settings, reports, metadata))
    for path in paths:
        logger.info("wrote %s", path)
    return paths
