import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cryptosignal import cli
from tests.fakes import FakeProvider, make_ohlcv

runner = CliRunner()


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the CLI against a temp config and a fake data provider."""
    config = tmp_path / "config.yaml"
    config.write_text(
        f"data:\n  data_dir: {tmp_path / 'data'}\n  min_rows: 300\n"
        f"artifacts_dir: {tmp_path / 'artifacts'}\nmodels_dir: {tmp_path / 'models'}\n"
        "backtest:\n  min_train_days: 300\n  retrain_every: 90\n"
        "model:\n  hgb_max_iter: 20\n"
    )
    monkeypatch.setenv("CRYPTOSIGNAL_CONFIG", str(config))
    fake = FakeProvider({"BTC-USD": make_ohlcv(seed=1), "ETH-USD": make_ohlcv(seed=2)})
    monkeypatch.setattr(cli, "YFinanceProvider", lambda: fake)
    return tmp_path


def test_version() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_config_prints_json(workdir: Path) -> None:
    result = runner.invoke(cli.app, ["config"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["backtest"]["min_train_days"] == 300


def test_ingest(workdir: Path) -> None:
    result = runner.invoke(cli.app, ["ingest", "--ticker", "btc-usd"])
    assert result.exit_code == 0, result.output
    assert "BTC-USD: 900 bars" in result.stdout
    assert (workdir / "data" / "raw" / "BTC-USD.csv").is_file()


def test_ingest_unknown_ticker_fails_cleanly(workdir: Path) -> None:
    result = runner.invoke(cli.app, ["ingest", "--ticker", "DOGE-USD"])
    assert result.exit_code == 1
