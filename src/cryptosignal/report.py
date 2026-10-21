"""Human-readable outputs: Markdown backtest report and model card."""

from datetime import UTC, datetime

from cryptosignal import __version__
from cryptosignal.backtest import BacktestReport
from cryptosignal.config import Settings
from cryptosignal.models import MODEL_LABELS, ModelMetadata


def _pct(x: float) -> str:
    return f"{x * 100:,.1f}%"


def _settings_sentence(report: BacktestReport) -> str:
    s = report.settings
    window = (
        "expanding" if s["window"] == "expanding" else f"rolling {s['rolling_window_days']}-day"
    )
    return (
        f"Walk-forward with an {window} training window (first fit on {s['min_train_days']} days), "
        f"models re-fitted every {s['retrain_every']} days, long/flat positions "
        f"(long when P(up) > {s['threshold']}), {s['cost_bps']:g} bps cost per position change."
    )


def ticker_section(report: BacktestReport) -> str:
    lines = [
        f"## {report.ticker}",
        "",
        f"Data {report.data_start} -> {report.data_end}; out-of-sample test "
        f"{report.test_start} -> {report.test_end} ({report.n_test} days, "
        f"{_pct(report.up_rate)} up days).",
        "",
        "### Next-day direction",
        "",
        "| Predictor | Accuracy | 95% CI | ROC-AUC | Predicts up |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in report.classification.items():
        auc = f"{m.roc_auc:.3f}" if m.roc_auc is not None else "-"
        lo, hi = m.accuracy_ci95
        lines.append(
            f"| {report.labels.get(name, name)} | {_pct(m.accuracy)} | "
            f"{_pct(lo)} - {_pct(hi)} | {auc} | {_pct(m.up_rate_predicted)} |"
        )
    lines += [
        "",
        "### Long/flat strategy, net of costs",
        "",
        "| Strategy | Total return | CAGR | Sharpe | Max drawdown | Exposure | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in report.strategies.items():
        lines.append(
            f"| {report.labels.get(name, name)} | {_pct(s.total_return)} | {_pct(s.cagr)} | "
            f"{s.sharpe:.2f} | {_pct(s.max_drawdown)} | {_pct(s.exposure)} | {s.trades} |"
        )
    return "\n".join(lines)


def render_report(reports: list[BacktestReport]) -> str:
    head = [
        "# Backtest report",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by cryptosignal {__version__}.",
        "",
        _settings_sentence(reports[0]) if reports else "",
        "Accuracy CIs use the normal approximation; an interval that contains the up-day",
        "rate means the model is not distinguishable from always predicting 'up'.",
        "",
    ]
    return "\n".join(head + [ticker_section(r) + "\n" for r in reports])


def render_model_card(
    settings: Settings,
    reports: list[BacktestReport],
    metadata: dict[str, ModelMetadata],
) -> str:
    kind = settings.model.kind
    lines = [
        f"# Model card - next-day direction classifier ({MODEL_LABELS[kind]})",
        "",
        "## Model details",
        "",
        "- **Task:** binary classification, P(close[t+1] > close[t]) from daily bars up to t.",
        f"- **Served model:** `{kind}` ({MODEL_LABELS[kind]}); also evaluated: "
        + ", ".join(f"`{m}`" for m in settings.backtest.models),
        f"- **Package:** cryptosignal {__version__}",
    ]
    for ticker, meta in metadata.items():
        lines.append(
            f"- **{ticker}:** version `{meta.version}`, trained {meta.trained_at} on "
            f"{meta.n_samples} days ({meta.data_start} -> {meta.data_end}), "
            f"scikit-learn {meta.sklearn_version}"
        )
    if metadata:
        features = next(iter(metadata.values())).feature_names
        lines.append(f"- **Features ({len(features)}):** " + ", ".join(f"`{f}`" for f in features))
    lines += [
        "",
        "## Intended use",
        "",
        "Educational and portfolio demonstration of a reproducible ML pipeline (ingestion,",
        "validation, leakage-free features, walk-forward evaluation, serving). It is **not**",
        "intended for trading decisions and is not financial advice.",
        "",
        "## Training data",
        "",
        "Daily OHLCV bars from Yahoo Finance (via `yfinance`), validated before use (schema,",
        "duplicates, gaps, non-positive prices...). The still-forming daily bar is dropped.",
        "",
        "## Evaluation (walk-forward, out-of-sample)",
        "",
    ]
    if reports:
        lines += [
            _settings_sentence(reports[0]),
            "",
            "| Ticker | Test period | Up-day rate | Accuracy | ROC-AUC | Sharpe (model) "
            "| Sharpe (buy & hold) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in reports:
            if kind not in r.classification:
                continue
            c, s = r.classification[kind], r.strategies[kind]
            auc = f"{c.roc_auc:.3f}" if c.roc_auc is not None else "-"
            lines.append(
                f"| {r.ticker} | {r.test_start} -> {r.test_end} | {_pct(r.up_rate)} | "
                f"{_pct(c.accuracy)} | {auc} | {s.sharpe:.2f} | "
                f"{r.strategies['buy_and_hold'].sharpe:.2f} |"
            )
        lines += ["", "Full tables: `report.md` / `report.json`."]
    lines += [
        "",
        "## Limitations",
        "",
        "- Daily crypto direction is close to a coin flip; expect accuracy near the up-day",
        "  rate and ROC-AUC near 0.5. Small edges are not statistically robust.",
        "- Costs are a flat per-trade charge; spreads, slippage in fast markets, funding and",
        "  taxes are not modelled. Trades are assumed to fill at the daily close.",
        "- Single data source (Yahoo Finance), no survivorship or delisting handling.",
        "- Hyper-parameters were fixed a priori, not tuned on the test period; comparing two",
        "  models on the same test period still carries some selection risk.",
        "- Non-stationary markets: past performance does not predict future results.",
    ]
    return "\n".join(lines) + "\n"
