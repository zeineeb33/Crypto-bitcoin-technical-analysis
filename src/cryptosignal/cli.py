"""Command-line interface: ``cryptosignal --help``."""

import json
from pathlib import Path
from typing import Annotated

import typer

from cryptosignal import __version__
from cryptosignal.config import Settings, load_settings
from cryptosignal.logging_utils import setup_logging

app = typer.Typer(
    help="Crypto technical-analysis pipeline and signal service.", no_args_is_help=True
)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="YAML config file (env vars still take precedence)."),
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


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("config")
def show_config() -> None:
    """Print the effective configuration as JSON."""
    typer.echo(json.dumps(_settings().model_dump(mode="json"), indent=2))
