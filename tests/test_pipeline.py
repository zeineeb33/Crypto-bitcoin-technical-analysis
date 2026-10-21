import json

from cryptosignal import pipeline
from cryptosignal.config import Settings
from tests.fakes import FakeProvider


def test_train_then_backtest_writes_artifacts(settings: Settings, provider: FakeProvider) -> None:
    models = pipeline.train(["BTC-USD"], settings, provider)
    assert (settings.models_dir / "BTC-USD" / "logreg.joblib").is_file()

    reports = pipeline.backtest(["BTC-USD", "ETH-USD"], settings, provider)

    payload = json.loads((settings.artifacts_dir / "report.json").read_text())
    assert set(payload["reports"]) == {"BTC-USD", "ETH-USD"}
    assert payload["validation"]["BTC-USD"]["ok"] is True
    btc = payload["reports"]["BTC-USD"]
    assert btc["classification"]["logreg"]["n"] == reports["BTC-USD"].n_test
    assert len(btc["equity"]["dates"]) == reports["BTC-USD"].n_test

    markdown = (settings.artifacts_dir / "report.md").read_text()
    assert "## BTC-USD" in markdown
    assert "Buy & hold" in markdown
    card = (settings.artifacts_dir / "model_card.md").read_text()
    assert models["BTC-USD"].metadata.version in card
    assert "not financial advice" in card
