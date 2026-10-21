import json
from datetime import date
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from cryptosignal.config import DataSettings
from cryptosignal.data import providers as providers_module
from cryptosignal.data.ingest import ingest, load_market_data
from cryptosignal.data.providers import (
    CSVProvider,
    DataProviderError,
    MarketDataProvider,
    YFinanceProvider,
    normalise_ohlcv,
)
from cryptosignal.data.validation import (
    DataValidationError,
    Severity,
    clean_ohlcv,
    validate_ohlcv,
)
from tests.fakes import TODAY, FakeProvider, make_ohlcv


def _yahoo_style(df: pd.DataFrame) -> pd.DataFrame:
    """Mimic the frame returned by yfinance: Title-case columns, tz-aware UTC index."""
    raw = df.rename(columns=str.title).assign(**{"Adj Close": df["close"]})
    raw.index = raw.index.tz_localize("UTC")
    raw.index.name = "Date"
    return raw


def _checks(report, severity: Severity) -> set[str]:  # type: ignore[no-untyped-def]
    return {i.check for i in report.issues if i.severity is severity}


# --- providers -------------------------------------------------------------------------


def test_normalise_converts_yahoo_frame(ohlcv: pd.DataFrame) -> None:
    out = normalise_ohlcv(_yahoo_style(ohlcv))
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.tz is None
    assert out.index.name == "date"
    pd.testing.assert_frame_equal(out, ohlcv, check_freq=False)


def test_normalise_rejects_missing_columns(ohlcv: pd.DataFrame) -> None:
    with pytest.raises(DataProviderError, match="missing columns"):
        normalise_ohlcv(ohlcv.drop(columns="volume"))


class _FakeTicker:
    responses: ClassVar[list[object]] = []

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, **kwargs: object) -> pd.DataFrame:
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, pd.DataFrame)
        return response


@pytest.fixture
def fake_yfinance(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTicker]:
    import yfinance

    _FakeTicker.responses = []
    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)
    return _FakeTicker


def test_yfinance_provider_retries_then_succeeds(
    fake_yfinance: type[_FakeTicker], ohlcv: pd.DataFrame
) -> None:
    fake_yfinance.responses = [ConnectionError("boom"), pd.DataFrame(), _yahoo_style(ohlcv)]
    sleeps: list[float] = []
    provider = YFinanceProvider(retries=3, backoff_seconds=1.0, sleep=sleeps.append)

    out = provider.fetch("BTC-USD", "max")

    assert len(out) == len(ohlcv)
    assert sleeps == [1.0, 2.0]  # exponential backoff
    assert isinstance(provider, MarketDataProvider)


def test_yfinance_provider_gives_up(fake_yfinance: type[_FakeTicker]) -> None:
    fake_yfinance.responses = [pd.DataFrame(), pd.DataFrame()]
    provider = YFinanceProvider(retries=2, sleep=lambda _: None)
    with pytest.raises(DataProviderError, match="could not fetch"):
        provider.fetch("NOPE-USD")


def test_yfinance_provider_rejects_bad_period() -> None:
    with pytest.raises(ValueError, match="invalid period"):
        YFinanceProvider().fetch("BTC-USD", "forever")


def test_csv_provider_round_trip(tmp_path, ohlcv: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
    store = CSVProvider(tmp_path)
    store.save("BTC-USD", ohlcv)
    out = store.fetch("BTC-USD")
    pd.testing.assert_frame_equal(out, ohlcv, check_freq=False, rtol=1e-9)


def test_csv_provider_missing_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(DataProviderError, match="run `cryptosignal ingest`"):
        CSVProvider(tmp_path).fetch("BTC-USD")


def test_provider_module_does_not_import_yfinance_eagerly() -> None:
    assert "yf" not in vars(providers_module)


# --- validation ------------------------------------------------------------------------


def test_clean_data_passes(ohlcv: pd.DataFrame) -> None:
    report = validate_ohlcv(ohlcv, ticker="BTC-USD", today=TODAY)
    assert report.ok
    assert report.issues == []
    assert report.end == "2023-12-31"


def test_clean_drops_duplicates_and_partial_bar(ohlcv: pd.DataFrame) -> None:
    today_bar = ohlcv.iloc[[-1]].set_axis([pd.Timestamp(TODAY)])
    dirty = pd.concat([ohlcv, ohlcv.iloc[[10]], today_bar]).sample(frac=1, random_state=0)

    cleaned, issues = clean_ohlcv(dirty, today=TODAY)

    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.duplicated().any()
    assert cleaned.index.max() < pd.Timestamp(TODAY)
    assert [i.check for i in issues] == ["duplicates"]


@pytest.mark.parametrize(
    ("mutate", "check"),
    [
        (
            lambda df: df.assign(close=df["close"].where(df.index != df.index[5], -1.0)),
            "non_positive_price",
        ),
        (
            lambda df: df.assign(open=df["open"].where(df.index != df.index[5], np.nan)),
            "missing_values",
        ),
        (
            lambda df: df.assign(volume=df["volume"].where(df.index != df.index[5], -5.0)),
            "negative_volume",
        ),
        (lambda df: df.assign(high=df["low"] * 0.5), "high_below_low"),
        (lambda df: df.drop(index=df.index[100:110]), "gaps"),
        (lambda df: df.iloc[::-1], "ordering"),
        (lambda df: pd.concat([df, df.iloc[[3]]]).sort_index(), "duplicates"),
        (lambda df: df.iloc[:50], "min_rows"),
        (lambda df: df.drop(columns="close"), "schema"),
    ],
)
def test_validation_errors(ohlcv: pd.DataFrame, mutate, check: str) -> None:  # type: ignore[no-untyped-def]
    report = validate_ohlcv(mutate(ohlcv), ticker="BTC-USD", today=TODAY)
    assert check in _checks(report, Severity.ERROR)
    with pytest.raises(DataValidationError, match=check):
        report.raise_for_errors()


def test_validation_warnings(ohlcv: pd.DataFrame) -> None:
    df = ohlcv.copy()
    df.iloc[3, df.columns.get_loc("volume")] = 0.0
    df = df.drop(index=df.index[200:202])  # 3-day hole: tolerated with max_gap_days=3
    df.iloc[400, df.columns.get_loc("close")] *= 3  # +200% candle
    report = validate_ohlcv(df, ticker="BTC-USD", today=date(2024, 2, 1))

    assert report.ok
    assert {"zero_volume", "gaps", "extreme_moves", "stale", "ohlc_consistency"} <= _checks(
        report, Severity.WARNING
    )


# --- ingestion -------------------------------------------------------------------------


def test_load_market_data_raises_on_bad_data(provider: FakeProvider) -> None:
    provider.frames["BTC-USD"] = provider.frames["BTC-USD"].iloc[:100]
    with pytest.raises(DataValidationError):
        load_market_data(provider, "BTC-USD", DataSettings(), today=TODAY)


def test_ingest_writes_csv_and_report(tmp_path, provider: FakeProvider) -> None:  # type: ignore[no-untyped-def]
    settings = DataSettings(data_dir=tmp_path)
    reports = ingest(["BTC-USD", "ETH-USD"], provider, settings, today=TODAY)

    assert set(reports) == {"BTC-USD", "ETH-USD"}
    stored = CSVProvider(tmp_path).fetch("ETH-USD")
    assert len(stored) == len(make_ohlcv(seed=2))
    summary = json.loads((tmp_path / "raw" / "BTC-USD.validation.json").read_text())
    assert summary["ok"] is True
