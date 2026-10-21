"""Walk-forward evaluation against naive baselines, with transaction costs.

Protocol (no look-ahead):

1. Rows are daily bars ``t`` with features known at the close of ``t`` and a label equal
   to the direction of ``close[t+1]``.
2. The test period starts after ``min_train_days`` labelled rows. Every ``retrain_every``
   rows the model is re-fitted on rows strictly before the block (expanding or rolling
   window) and predicts the block. The newest training label uses ``close[s]`` where ``s``
   is the first test row, which is known when the prediction for ``s`` is made.
3. Positions are long (1) or flat (0), decided at the close of ``t`` and held until the
   close of ``t+1``. Every change of position pays ``cost_bps`` (fees + slippage).
4. Every strategy and baseline is scored on exactly the same test days.
"""

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from cryptosignal.config import BacktestSettings, FeatureSettings, ModelKind, ModelSettings
from cryptosignal.features import compute_indicators, make_dataset
from cryptosignal.models import MODEL_LABELS, make_model, predict_up_proba

logger = logging.getLogger(__name__)

BASELINE_LABELS: dict[str, str] = {
    "buy_and_hold": "Buy & hold (always long)",
    "sma_crossover": "SMA crossover",
    "macd_crossover": "MACD crossover (notebook rule)",
    "momentum": "Momentum (repeat today's direction)",
}


def walk_forward_proba(
    features: pd.DataFrame,
    target: pd.Series,
    model_factory: Callable[[], Any],
    *,
    min_train: int,
    step: int,
    window: str = "expanding",
    rolling_size: int | None = None,
) -> pd.Series:
    """Out-of-sample P(up) for every row from ``min_train`` onwards."""
    n = len(features)
    if n <= min_train:
        raise ValueError(f"need more than {min_train} labelled rows, got {n}")
    proba = np.full(n, np.nan)
    n_folds = 0
    for start in range(min_train, n, step):
        stop = min(start + step, n)
        lo = max(0, start - rolling_size) if window == "rolling" and rolling_size else 0
        model = model_factory()
        model.fit(features.iloc[lo:start], target.iloc[lo:start])
        proba[start:stop] = predict_up_proba(model, features.iloc[start:stop])
        n_folds += 1
    logger.debug("walk-forward: %d folds, %d test rows", n_folds, n - min_train)
    return pd.Series(proba[min_train:], index=features.index[min_train:], name="proba_up")


def strategy_returns(position: pd.Series, forward_return: pd.Series, cost_bps: float) -> pd.Series:
    """Daily net returns of a long/flat strategy. Entering on day one also pays costs."""
    turnover = position.diff().abs().fillna(position.abs())
    return position * forward_return - turnover * cost_bps / 10_000


@dataclass(frozen=True)
class StrategyMetrics:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    exposure: float
    trades: int


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    accuracy_ci95: tuple[float, float]
    roc_auc: float | None
    up_rate_predicted: float
    n: int


def strategy_metrics(
    returns: pd.Series, position: pd.Series, periods_per_year: int = 365
) -> StrategyMetrics:
    equity = (1 + returns).cumprod()
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity.to_numpy()]))[1:]
    years = len(returns) / periods_per_year
    std = float(returns.std(ddof=1))
    final = float(equity.iloc[-1])
    return StrategyMetrics(
        total_return=final - 1,
        cagr=final ** (1 / years) - 1 if final > 0 else -1.0,
        volatility=std * np.sqrt(periods_per_year),
        sharpe=float(returns.mean()) / std * np.sqrt(periods_per_year) if std > 0 else 0.0,
        max_drawdown=float((equity.to_numpy() / peak - 1).min()),
        exposure=float(position.mean()),
        trades=int(position.diff().abs().fillna(position.abs()).sum()),
    )


def classification_metrics(
    y_true: pd.Series, predicted_up: pd.Series, proba: pd.Series | None = None
) -> ClassificationMetrics:
    n = len(y_true)
    acc = float(accuracy_score(y_true, predicted_up))
    half_width = 1.96 * np.sqrt(acc * (1 - acc) / n)
    auc = None
    if proba is not None and y_true.nunique() == 2:
        auc = float(roc_auc_score(y_true, proba))
    return ClassificationMetrics(
        accuracy=acc,
        accuracy_ci95=(acc - half_width, acc + half_width),
        roc_auc=auc,
        up_rate_predicted=float(predicted_up.mean()),
        n=n,
    )


@dataclass
class BacktestReport:
    ticker: str
    generated_at: str
    data_start: str
    data_end: str
    test_start: str
    test_end: str
    n_test: int
    up_rate: float
    settings: dict[str, Any]
    models: list[str]
    classification: dict[str, ClassificationMetrics]
    strategies: dict[str, StrategyMetrics]
    equity: pd.DataFrame = field(repr=False)
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self, include_equity: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            k: v
            for k, v in asdict(self).items()
            if k not in {"equity", "classification", "strategies"}
        }
        out["classification"] = {k: asdict(v) for k, v in self.classification.items()}
        out["strategies"] = {k: asdict(v) for k, v in self.strategies.items()}
        if include_equity:
            out["equity"] = {
                "dates": [d.strftime("%Y-%m-%d") for d in self.equity.index],
                **{c: self.equity[c].round(6).tolist() for c in self.equity.columns},
            }
        return out


def run_backtest(
    ohlcv: pd.DataFrame,
    *,
    ticker: str,
    models: tuple[ModelKind, ...] | list[ModelKind],
    features_cfg: FeatureSettings,
    model_cfg: ModelSettings,
    cfg: BacktestSettings,
) -> BacktestReport:
    data = make_dataset(ohlcv, features_cfg).labelled()
    indicators = compute_indicators(ohlcv, features_cfg).loc[data.features.index]
    test_index = data.features.index[cfg.min_train_days :]
    if len(test_index) < 30:
        raise ValueError(
            f"{ticker}: only {len(test_index)} test days after a {cfg.min_train_days}-day "
            "training window - fetch more history or lower backtest.min_train_days"
        )
    y = data.target.loc[test_index]
    fwd = data.forward_return.loc[test_index]

    positions: dict[str, pd.Series] = {}
    probas: dict[str, pd.Series] = {}
    for kind in models:
        probas[kind] = walk_forward_proba(
            data.features,
            data.target,
            lambda kind=kind: make_model(kind, model_cfg),  # type: ignore[misc]
            min_train=cfg.min_train_days,
            step=cfg.retrain_every,
            window=cfg.window,
            rolling_size=cfg.rolling_window_days,
        )
        positions[kind] = (probas[kind] > cfg.threshold).astype(float)

    sma_fast = ohlcv["close"].rolling(cfg.sma_fast).mean().loc[test_index]
    sma_slow = ohlcv["close"].rolling(cfg.sma_slow).mean().loc[test_index]
    positions["buy_and_hold"] = pd.Series(1.0, index=test_index)
    positions["sma_crossover"] = (sma_fast > sma_slow).astype(float)
    positions["macd_crossover"] = (
        indicators["macd"].loc[test_index] > indicators["macd_signal"].loc[test_index]
    ).astype(float)
    positions["momentum"] = (ohlcv["close"].pct_change().loc[test_index] > 0).astype(float)

    classification: dict[str, ClassificationMetrics] = {}
    strategies: dict[str, StrategyMetrics] = {}
    equity: dict[str, pd.Series] = {}
    for name, position in positions.items():
        classification[name] = classification_metrics(y, position, probas.get(name))
        net = strategy_returns(position, fwd, cfg.cost_bps)
        strategies[name] = strategy_metrics(net, position, cfg.periods_per_year)
        equity[name] = (1 + net).cumprod()

    labels = {k: MODEL_LABELS[k] for k in models} | BASELINE_LABELS
    labels["sma_crossover"] = f"SMA {cfg.sma_fast}/{cfg.sma_slow} crossover"
    # Equity is valued at the close of t+1, so shift the dates by one bar for plotting.
    equity_df = pd.DataFrame(equity)
    equity_df.index = ohlcv.index[ohlcv.index.get_indexer(test_index) + 1]

    report = BacktestReport(
        ticker=ticker,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        data_start=str(ohlcv.index[0].date()),
        data_end=str(ohlcv.index[-1].date()),
        test_start=str(test_index[0].date()),
        test_end=str(test_index[-1].date()),
        n_test=len(test_index),
        up_rate=float(y.mean()),
        settings=cfg.model_dump(mode="json"),
        models=list(models),
        classification=classification,
        strategies=strategies,
        equity=equity_df,
        labels=labels,
    )
    logger.info(
        "%s backtest %s -> %s (%d days): %s",
        ticker,
        report.test_start,
        report.test_end,
        report.n_test,
        ", ".join(f"{m} acc={classification[m].accuracy:.3f}" for m in models),
    )
    return report
