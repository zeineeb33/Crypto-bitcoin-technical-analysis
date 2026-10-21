"""Command-line interface: ``cryptosignal --help``."""

import json
import logging
import os
from pathlib import Path
from typing import Annotated

import typer

from cryptosignal import __version__, pipeline
from cryptosignal.config import (
    CONFIG_ENV_VAR,
    ModelKind,
    Settings,
    load_settings,
    normalise_ticker,
)
from cryptosignal.data.ingest import ingest as ingest_data
from cryptosignal.data.ingest import load_market_data
from cryptosignal.data.providers import CSVProvider, DataProviderError, YFinanceProvider
from cryptosignal.data.validation import DataValidationError
from cryptosignal.features import make_dataset
from cryptosignal.models import TrainedModel
from cryptosignal.runtime import configure_runtime

logger = logging.getLogger("cryptosignal")

app = typer.Typer(
    help="Crypto technical-analysis pipeline and signal service.", no_args_is_help=True
)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="YAML config file (env vars still take precedence)."),
]
TickersOption = Annotated[
    list[str] | None,
    typer.Option("--ticker", "-t", help="Ticker(s) to process; defaults to the configured list."),
]
ModelOption = Annotated[
    ModelKind | None,
    typer.Option("--model", "-m", help="Model kind; defaults to model.kind from the config."),
]
PIPELINE_ERRORS = (DataProviderError, DataValidationError, FileNotFoundError, ValueError)


class _State:
    settings: Settings | None = None


@app.callback()
def main(config: ConfigOption = None) -> None:
    settings = load_settings(config)
    if config is not None:  # let the API factory (`serve`) pick the same file up
        os.environ[CONFIG_ENV_VAR] = str(config.resolve())
    configure_runtime(settings)
    _State.settings = settings


def _settings() -> Settings:
    if _State.settings is None:  # pragma: no cover - callback always runs first
        _State.settings = load_settings()
    return _State.settings


def _tickers(tickers: list[str] | None) -> list[str]:
    return [normalise_ticker(t) for t in tickers] if tickers else list(_settings().tickers)


def _fail(exc: Exception) -> typer.Exit:
    logger.error("%s", exc)
    return typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("config")
def show_config() -> None:
    """Print the effective configuration as JSON."""
    typer.echo(json.dumps(_settings().model_dump(mode="json"), indent=2))


@app.command()
def ingest(tickers: TickersOption = None) -> None:
    """Download daily bars, validate them and store them under data/raw/."""
    settings = _settings()
    try:
        reports = ingest_data(_tickers(tickers), YFinanceProvider(), settings.data)
    except PIPELINE_ERRORS as exc:
        raise _fail(exc) from exc
    for ticker, report in reports.items():
        typer.echo(
            f"{ticker}: {report.n_rows} bars {report.start} -> {report.end} "
            f"({len(report.warnings)} warnings)"
        )


def _local_data() -> CSVProvider:
    return CSVProvider(_settings().data.data_dir)


@app.command()
def train(tickers: TickersOption = None, model: ModelOption = None) -> None:
    """Fit the model on all stored history and save it under models/."""
    try:
        models = pipeline.train(_tickers(tickers), _settings(), _local_data(), kind=model)
    except PIPELINE_ERRORS as exc:
        raise _fail(exc) from exc
    for ticker, trained in models.items():
        typer.echo(f"{ticker}: {trained.metadata.version} ({trained.metadata.n_samples} samples)")


@app.command()
def backtest(tickers: TickersOption = None) -> None:
    """Walk-forward backtest vs baselines; writes artifacts/report.{json,md} and model_card.md."""
    settings = _settings()
    try:
        reports = pipeline.backtest(_tickers(tickers), settings, _local_data())
    except PIPELINE_ERRORS as exc:
        raise _fail(exc) from exc
    for ticker, report in reports.items():
        for name in report.models:
            c, s = report.classification[name], report.strategies[name]
            typer.echo(
                f"{ticker} {name}: accuracy {c.accuracy:.3f} (up-day rate {report.up_rate:.3f}), "
                f"AUC {c.roc_auc or float('nan'):.3f}, Sharpe {s.sharpe:.2f} "
                f"(buy & hold {report.strategies['buy_and_hold'].sharpe:.2f})"
            )
    typer.echo(f"report: {settings.artifacts_dir / 'report.md'}")


@app.command()
def run(tickers: TickersOption = None, fetch: bool = True) -> None:
    """Full pipeline: ingest -> train -> backtest (use --no-fetch to reuse data/raw)."""
    if fetch:
        ingest(tickers)
    train(tickers)
    backtest(tickers)


@app.command()
def predict(tickers: TickersOption = None, model: ModelOption = None) -> None:
    """Print the latest signal from the saved model and stored data as JSON."""
    settings = _settings()
    kind = model or settings.model.kind
    out = {}
    try:
        for ticker in _tickers(tickers):
            trained = TrainedModel.load(settings.models_dir, ticker, kind)
            ohlcv, _ = load_market_data(_local_data(), ticker, settings.data)
            latest = make_dataset(ohlcv, settings.features).latest()
            proba = float(trained.predict_up_proba(latest)[0])
            out[ticker] = {
                "as_of": str(latest.index[-1].date()),
                "probability_up": round(proba, 4),
                "signal": "long" if proba > settings.backtest.threshold else "flat",
                "model_version": trained.metadata.version,
            }
    except PIPELINE_ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(json.dumps(out, indent=2))


@app.command()
def serve(
    host: Annotated[
        str, typer.Option(help="Bind address (use 0.0.0.0 in containers).")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(envvar="PORT", help="Port; reads $PORT.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (dev).")] = False,
) -> None:
    """Serve the API and dashboard with uvicorn."""
    import uvicorn  # noqa: PLC0415 - only needed for this command

    uvicorn.run(
        "cryptosignal.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level=_settings().log_level.lower(),
    )
