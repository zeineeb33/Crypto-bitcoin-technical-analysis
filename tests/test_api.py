from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from cryptosignal.api.app import create_app
from cryptosignal.config import Settings
from tests.fakes import FakeProvider


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2024, 1, 1, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def client(settings: Settings, provider: FakeProvider) -> TestClient:
    return TestClient(create_app(settings, provider))


def test_health_does_not_touch_provider(client: TestClient, provider: FakeProvider) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["tickers"]["BTC-USD"]["ready"] is False
    assert provider.calls == 0


def test_signal_indicators_backtest(client: TestClient, provider: FakeProvider) -> None:
    signal = client.get("/signal/btc-usd").json()
    assert signal["ticker"] == "BTC-USD"
    assert signal["as_of"] == "2023-12-31"
    assert signal["prediction_for"] == "2024-01-01"
    assert signal["signal"] in {"long", "flat"}
    assert signal["model_version"].startswith("logreg-")

    ind = client.get("/indicators/BTC-USD?days=60").json()
    assert len(ind["points"]) == 60
    assert ind["points"][-1]["macd"] is not None

    bt = client.get("/backtest/BTC-USD").json()
    assert bt["models"] == ["logreg"]
    assert "buy_and_hold" in bt["strategies"]
    assert len(bt["equity"]["dates"]) == bt["n_test"]
    assert provider.calls == 1  # one fetch, then served from cache
    assert client.get("/health").json()["tickers"]["BTC-USD"]["ready"] is True


def test_unknown_ticker_is_404(client: TestClient) -> None:
    assert client.get("/signal/DOGE-USD").status_code == 404


def test_provider_down_returns_503_with_cooldown(
    client: TestClient, provider: FakeProvider
) -> None:
    provider.fail = True
    first = client.get("/signal/BTC-USD")
    assert first.status_code == 503
    assert int(first.headers["Retry-After"]) > 0
    assert client.get("/signal/BTC-USD").status_code == 503
    assert provider.calls == 1  # cool-down: no hammering of the provider
    assert client.get("/health").status_code == 200


def test_stale_snapshot_served_when_refresh_fails(
    settings: Settings, provider: FakeProvider
) -> None:
    clock = Clock()
    app = create_app(settings, provider)
    app.state.service.clock = clock
    client = TestClient(app)
    assert client.get("/signal/BTC-USD").json()["stale"] is False

    provider.fail = True
    clock.now += timedelta(minutes=settings.serve.refresh_minutes + 1)
    response = client.get("/signal/BTC-USD")
    assert response.status_code == 200
    assert response.json()["stale"] is True
    assert client.get("/health").json()["tickers"]["BTC-USD"]["last_error"]


def test_dashboard_and_docs_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "lightweight-charts" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert "/signal/{ticker}" in client.get("/openapi.json").json()["paths"]
