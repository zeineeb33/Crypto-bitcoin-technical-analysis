from pathlib import Path

import pytest
from pydantic import ValidationError

from cryptosignal.config import Settings, load_settings, normalise_ticker


def test_defaults_are_valid() -> None:
    settings = Settings()
    assert settings.tickers == ["BTC-USD", "ETH-USD"]
    assert settings.backtest.cost_bps == 10.0


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("tickers: [btc-usd]\nbacktest:\n  cost_bps: 5\n  retrain_every: 10\n")
    monkeypatch.setenv("CRYPTOSIGNAL_BACKTEST__RETRAIN_EVERY", "60")

    settings = load_settings(config)

    assert settings.tickers == ["BTC-USD"]
    assert settings.backtest.cost_bps == 5
    assert settings.backtest.retrain_every == 60  # env wins over the file


def test_config_env_var_points_to_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("serve:\n  refresh_minutes: 15\n")
    monkeypatch.setenv("CRYPTOSIGNAL_CONFIG", str(config))
    assert load_settings().serve.refresh_minutes == 15


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.yaml")


@pytest.mark.parametrize("bad", ["", "BTC USD", "../etc", "x" * 30])
def test_invalid_tickers_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid ticker"):
        normalise_ticker(bad)


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(backtest={"cost": 3})  # type: ignore[arg-type]


def test_sma_baseline_windows_validated() -> None:
    with pytest.raises(ValidationError):
        Settings(backtest={"sma_fast": 200, "sma_slow": 50})  # type: ignore[arg-type]
