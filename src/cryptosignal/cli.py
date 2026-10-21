"""Command-line interface: ``cryptosignal --help``."""

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from cryptosignal import __version__
from cryptosignal.config import Settings, load_settings, normalise_ticker
from cryptosignal.data.ingest import ingest as ingest_data
from cryptosignal.data.providers import DataProviderError, YFinanceProvider
from cryptosignal.data.validation import DataValidationError
from cryptosignal.logging_utils import setup_logging

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


class _State:
    settings: Settings | None = None


@app.callback()
def main(config: ConfigOption = None) -> None:
    settings = load_settings(config)
    setup_logging(settings.log_level)
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
    except (DataProviderError, DataValidationError) as exc:
        raise _fail(exc) from exc
    for ticker, report in reports.items():
        typer.echo(
            f"{ticker}: {report.n_rows} bars {report.start} -> {report.end} "
            f"({len(report.warnings)} warnings)"
        )
