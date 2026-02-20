"""
Request-scoped cost telemetry helpers.

Telemetry is enabled by attaching a CostCollector via contextvars.
Providers read active collector/stage and emit usage records automatically.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

from vanna_kg.config.pricing import PRICING_VERSION
from vanna_kg.types.results import (
    CostBreakdown,
    CostDebugReport,
    CostUsageRecord,
    StageCostBreakdown,
)

_COLLECTOR: ContextVar[CostCollector | None] = ContextVar(
    "vanna_cost_collector",
    default=None,
)
_STAGE: ContextVar[str] = ContextVar("vanna_cost_stage", default="unknown")


class CostCollector:
    """Accumulates provider usage records for one request."""

    def __init__(self, *, warn_threshold_usd: float | None = None) -> None:
        self._records: list[CostUsageRecord] = []
        self._warn_threshold_usd = warn_threshold_usd

    def add(self, record: CostUsageRecord) -> None:
        """Add one usage record."""
        self._records.append(record)

    def summary(self) -> CostDebugReport:
        """Build aggregate report across all records."""
        by_stage: dict[str, StageCostBreakdown] = {}
        warnings: list[str] = []

        total_calls = len(self._records)
        total_input = 0
        total_output = 0
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0

        for record in self._records:
            total_input += record.input_tokens
            total_output += record.output_tokens
            total_tokens += record.total_tokens
            total_cost += record.estimated_cost_usd
            total_latency += record.latency_ms

            stage = by_stage.setdefault(record.stage, StageCostBreakdown(stage=record.stage))
            stage.calls += 1
            stage.input_tokens += record.input_tokens
            stage.output_tokens += record.output_tokens
            stage.total_tokens += record.total_tokens
            stage.estimated_cost_usd += record.estimated_cost_usd
            stage.total_latency_ms += record.latency_ms

            if record.metadata.get("pricing_found") is False:
                warnings.append(
                    f"Missing pricing for model '{record.model}' in stage '{record.stage}'. "
                    "Cost shown as 0.0 for those calls."
                )

        if self._warn_threshold_usd is not None and total_cost >= self._warn_threshold_usd:
            warnings.append(
                f"Estimated request cost ${total_cost:.6f} exceeded threshold "
                f"${self._warn_threshold_usd:.6f}."
            )

        breakdown = CostBreakdown(
            total_calls=total_calls,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
            total_estimated_cost_usd=total_cost,
            total_latency_ms=total_latency,
            by_stage=sorted(by_stage.values(), key=lambda s: s.estimated_cost_usd, reverse=True),
        )
        return CostDebugReport(
            enabled=True,
            pricing_version=PRICING_VERSION,
            breakdown=breakdown,
            warnings=sorted(set(warnings)),
        )


@contextmanager
def telemetry_collector(collector: CostCollector | None):
    """Set active request collector for provider instrumentation."""
    token = _COLLECTOR.set(collector)
    try:
        yield
    finally:
        _COLLECTOR.reset(token)


@contextmanager
def telemetry_stage(stage: str):
    """Set pipeline stage label for provider instrumentation."""
    token = _STAGE.set(stage)
    try:
        yield
    finally:
        _STAGE.reset(token)


def current_stage() -> str:
    """Return currently active telemetry stage label."""
    return _STAGE.get()


def record_usage(record: CostUsageRecord) -> None:
    """Add record to active collector if telemetry is enabled."""
    collector = _COLLECTOR.get()
    if collector is not None:
        collector.add(record)
