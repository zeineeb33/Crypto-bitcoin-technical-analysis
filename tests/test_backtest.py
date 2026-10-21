import numpy as np
import pandas as pd
import pytest

from cryptosignal.backtest import (
    classification_metrics,
    run_backtest,
    strategy_metrics,
    strategy_returns,
    walk_forward_proba,
)
from cryptosignal.config import BacktestSettings, FeatureSettings, ModelSettings
from cryptosignal.features import make_dataset
from cryptosignal.models import make_model
from tests.fakes import make_ohlcv, perturb_after

FEATURES = FeatureSettings()
MODEL = ModelSettings(hgb_max_iter=30)
SMALL = BacktestSettings(min_train_days=300, retrain_every=60)


def _run(ohlcv: pd.DataFrame, models=("logreg",), cfg: BacktestSettings = SMALL):  # type: ignore[no-untyped-def]
    return run_backtest(
        ohlcv, ticker="TEST", models=models, features_cfg=FEATURES, model_cfg=MODEL, cfg=cfg
    )


def test_strategy_returns_charge_costs_on_position_changes() -> None:
    position = pd.Series([1.0, 1.0, 0.0, 1.0])
    fwd = pd.Series([0.01, 0.02, 0.03, -0.01])
    net = strategy_returns(position, fwd, cost_bps=10)
    expected = [0.01 - 0.001, 0.02, 0.0 - 0.001, -0.01 - 0.001]
    np.testing.assert_allclose(net.to_numpy(), expected)


def test_strategy_metrics_known_values() -> None:
    returns = pd.Series([0.1, -0.5, 0.2])
    m = strategy_metrics(returns, pd.Series([1.0, 1.0, 1.0]), periods_per_year=365)
    assert m.total_return == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert m.max_drawdown == pytest.approx(-0.5)
    assert m.exposure == 1.0
    assert m.trades == 1


def test_drawdown_counts_losses_from_initial_capital() -> None:
    m = strategy_metrics(pd.Series([-0.2, 0.1]), pd.Series([1.0, 1.0]))
    assert m.max_drawdown == pytest.approx(-0.2)


def test_classification_metrics() -> None:
    y = pd.Series([1, 0, 1, 1])
    pred = pd.Series([1.0, 0.0, 0.0, 1.0])
    m = classification_metrics(y, pred, proba=pd.Series([0.9, 0.1, 0.4, 0.8]))
    assert m.accuracy == 0.75
    assert m.roc_auc == 1.0
    assert m.accuracy_ci95[0] < 0.75 < m.accuracy_ci95[1]


def test_walk_forward_only_trains_on_the_past() -> None:
    """Record every fit: training rows must all precede the rows being predicted."""
    features = pd.DataFrame({"x": np.arange(100.0)}, index=pd.date_range("2020", periods=100))
    target = pd.Series(np.arange(100) % 2, index=features.index)
    fits: list[tuple[pd.Timestamp, int]] = []

    class Spy:
        classes_ = np.array([0, 1])

        def fit(self, x: pd.DataFrame, y: pd.Series) -> "Spy":
            self.last_train = x.index[-1]
            fits.append((x.index[-1], len(x)))
            return self

        def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
            assert x.index[0] > self.last_train
            return np.tile([0.5, 0.5], (len(x), 1))

    proba = walk_forward_proba(features, target, Spy, min_train=40, step=25)
    assert proba.index[0] == features.index[40]
    assert len(proba) == 60
    assert [n for _, n in fits] == [40, 65, 90]  # expanding window

    walk_forward_proba(
        features, target, Spy, min_train=40, step=25, window="rolling", rolling_size=30
    )
    assert [n for _, n in fits[3:]] == [30, 30, 30]


def test_walk_forward_predictions_unaffected_by_future_data() -> None:
    """End-to-end leakage guard: rewriting the future must not change past predictions."""
    ohlcv = make_ohlcv(seed=3)
    cut = 650
    dates = ohlcv.index[: cut + 1]

    def proba(frame: pd.DataFrame) -> pd.Series:
        data = make_dataset(frame, FEATURES).labelled()
        out = walk_forward_proba(
            data.features,
            data.target,
            lambda: make_model("hgb", MODEL),
            min_train=300,
            step=60,
        )
        return out[out.index.isin(dates)]

    pd.testing.assert_series_equal(proba(ohlcv), proba(perturb_after(ohlcv, cut)))


def test_backtest_report_on_random_walk_is_close_to_chance() -> None:
    report = _run(make_ohlcv(n=1200, seed=11), models=("logreg", "hgb"))

    assert report.n_test == len(report.equity)
    assert set(report.strategies) == {
        "logreg",
        "hgb",
        "buy_and_hold",
        "sma_crossover",
        "macd_crossover",
        "momentum",
    }
    for kind in ("logreg", "hgb"):
        c = report.classification[kind]
        assert 0.4 < c.accuracy < 0.6  # no signal in a random walk
        assert c.roc_auc is not None
    assert report.strategies["buy_and_hold"].exposure == 1.0
    assert report.strategies["buy_and_hold"].trades == 1
    assert report.equity.index[0] > pd.Timestamp(report.test_start)

    payload = report.to_dict()
    assert len(payload["equity"]["dates"]) == report.n_test
    assert payload["classification"]["logreg"]["n"] == report.n_test


def test_backtest_detects_real_signal() -> None:
    """Positive control: with strongly autocorrelated returns the model must beat chance."""
    rng = np.random.default_rng(5)
    n = 1200
    ret = np.zeros(n)
    for t in range(1, n):
        ret[t] = 0.6 * ret[t - 1] + rng.normal(0, 0.02)
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2020-01-01", periods=n, freq="D").as_unit("ns")
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.lognormal(20, 0.3, n),
        },
        index=idx,
    )
    report = _run(ohlcv)
    assert report.classification["logreg"].accuracy > 0.65
    assert report.classification["logreg"].roc_auc > 0.7  # type: ignore[operator]


def test_backtest_requires_enough_history() -> None:
    with pytest.raises(ValueError, match="test days"):
        _run(make_ohlcv(n=340))
