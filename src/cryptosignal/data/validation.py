"""Data quality gate for daily OHLCV bars.

``clean_ohlcv`` applies the few *safe* fixes (sort, de-duplicate, drop the still-forming
bar); ``validate_ohlcv`` never mutates data and reports every issue it finds. Errors stop
the pipeline; warnings are logged and kept in the report.
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from cryptosignal.data.providers import OHLCV_COLUMNS

logger = logging.getLogger(__name__)

EXTREME_DAILY_MOVE = 0.5  # |return| above 50% in a day is suspicious, even for crypto
STALE_AFTER_DAYS = 3


class Severity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Issue:
    check: str
    severity: Severity
    message: str
    count: int = 0


@dataclass
class ValidationReport:
    ticker: str
    n_rows: int
    start: str | None
    end: str | None
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise DataValidationError(self)


class DataValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        details = "; ".join(f"{i.check}: {i.message}" for i in report.errors)
        super().__init__(f"{report.ticker}: data validation failed - {details}")


def utc_today() -> date:
    return datetime.now(UTC).date()


def clean_ohlcv(
    df: pd.DataFrame, *, drop_incomplete_bar: bool = True, today: date | None = None
) -> tuple[pd.DataFrame, list[Issue]]:
    """Sort, drop duplicate dates (keep the latest), all-NaN rows and today's partial bar."""
    issues: list[Issue] = []
    out = df.sort_index()

    duplicated = out.index.duplicated(keep="last")
    if duplicated.any():
        n = int(duplicated.sum())
        issues.append(Issue("duplicates", Severity.WARNING, f"dropped {n} duplicate dates", n))
        out = out[~duplicated]

    empty_rows = out.isna().all(axis=1)
    if empty_rows.any():
        n = int(empty_rows.sum())
        issues.append(Issue("empty_rows", Severity.WARNING, f"dropped {n} all-NaN rows", n))
        out = out[~empty_rows]

    if drop_incomplete_bar and len(out):
        cutoff = pd.Timestamp(today or utc_today())
        partial = out.index >= cutoff
        if partial.any():
            logger.info("dropping %d incomplete bar(s) dated >= %s", partial.sum(), cutoff.date())
            out = out[~partial]
    return out, issues


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    ticker: str,
    min_rows: int = 400,
    max_gap_days: int = 3,
    today: date | None = None,
) -> ValidationReport:
    """Run schema, completeness and sanity checks. Never mutates ``df``."""
    report = ValidationReport(
        ticker=ticker,
        n_rows=len(df),
        start=str(df.index.min().date()) if len(df) else None,
        end=str(df.index.max().date()) if len(df) else None,
    )
    add = report.issues.append

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        add(Issue("schema", Severity.ERROR, f"missing columns {missing}", len(missing)))
        return report
    if not isinstance(df.index, pd.DatetimeIndex):
        add(Issue("schema", Severity.ERROR, "index is not a DatetimeIndex"))
        return report
    if len(df) < min_rows:
        add(Issue("min_rows", Severity.ERROR, f"{len(df)} rows < required {min_rows}", len(df)))
    if len(df) == 0:
        return report

    if not df.index.is_monotonic_increasing:
        add(Issue("ordering", Severity.ERROR, "index is not sorted in increasing order"))
    if (n := int(df.index.duplicated().sum())) > 0:
        add(Issue("duplicates", Severity.ERROR, f"{n} duplicate dates", n))

    prices = df[["open", "high", "low", "close"]]
    if (n := int(prices.isna().any(axis=1).sum())) > 0:
        add(Issue("missing_values", Severity.ERROR, f"{n} rows with missing prices", n))
    if (n := int((prices <= 0).any(axis=1).sum())) > 0:
        add(Issue("non_positive_price", Severity.ERROR, f"{n} rows with price <= 0", n))
    if (n := int((df["volume"] < 0).sum())) > 0:
        add(Issue("negative_volume", Severity.ERROR, f"{n} rows with volume < 0", n))
    if (n := int((df["volume"] == 0).sum())) > 0:
        add(Issue("zero_volume", Severity.WARNING, f"{n} rows with zero volume", n))
    if (n := int((df["high"] < df["low"]).sum())) > 0:
        add(Issue("high_below_low", Severity.ERROR, f"{n} rows with high < low", n))

    body_high = np.maximum(df["open"], df["close"])
    body_low = np.minimum(df["open"], df["close"])
    inconsistent = (df["high"] < body_high) | (df["low"] > body_low)
    if (n := int(inconsistent.sum())) > 0:
        add(Issue("ohlc_consistency", Severity.WARNING, f"{n} bars outside their high/low", n))

    deltas = df.index.to_series().diff().dt.days.dropna()
    gaps = deltas[deltas > 1]
    if len(gaps):
        largest = int(gaps.max())
        severity = Severity.ERROR if largest > max_gap_days else Severity.WARNING
        msg = f"{len(gaps)} calendar gaps, largest {largest} days (max allowed {max_gap_days})"
        add(Issue("gaps", severity, msg, len(gaps)))

    moves = df["close"].pct_change().abs()
    if (n := int((moves > EXTREME_DAILY_MOVE).sum())) > 0:
        msg = f"{n} daily moves above {EXTREME_DAILY_MOVE:.0%}"
        add(Issue("extreme_moves", Severity.WARNING, msg, n))

    age = ((today or utc_today()) - df.index.max().date()).days
    if age > STALE_AFTER_DAYS:
        add(Issue("stale", Severity.WARNING, f"latest bar is {age} days old", age))

    for issue in report.issues:
        log = logger.error if issue.severity is Severity.ERROR else logger.warning
        log("%s [%s] %s", ticker, issue.check, issue.message)
    return report
