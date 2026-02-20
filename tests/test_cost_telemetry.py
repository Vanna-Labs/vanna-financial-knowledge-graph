"""Tests for request-scoped cost telemetry aggregation."""

from vanna_kg.types.results import CostUsageRecord
from vanna_kg.utils.cost_telemetry import CostCollector


def test_cost_collector_aggregates_by_stage() -> None:
    """Collector should aggregate totals and per-stage metrics."""
    collector = CostCollector()
    collector.add(
        CostUsageRecord(
            provider="openai",
            model="gpt-5-mini",
            operation="generate_structured",
            stage="decomposition",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            estimated_cost_usd=0.001,
            latency_ms=15,
            estimated=False,
        )
    )
    collector.add(
        CostUsageRecord(
            provider="openai",
            model="text-embedding-3-large",
            operation="embed",
            stage="research_resolution",
            input_tokens=50,
            output_tokens=0,
            total_tokens=50,
            estimated_cost_usd=0.0001,
            latency_ms=5,
            estimated=True,
        )
    )

    report = collector.summary()
    assert report.enabled is True
    assert report.breakdown.total_calls == 2
    assert report.breakdown.total_tokens == 170
    assert report.breakdown.total_input_tokens == 150
    assert report.breakdown.total_output_tokens == 20
    assert len(report.breakdown.by_stage) == 2
    assert report.breakdown.by_stage[0].stage == "decomposition"


def test_cost_collector_warns_on_threshold() -> None:
    """Collector should include warning when threshold is exceeded."""
    collector = CostCollector(warn_threshold_usd=0.0005)
    collector.add(
        CostUsageRecord(
            provider="openai",
            model="gpt-5-mini",
            operation="generate",
            stage="synthesis_final",
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
            estimated_cost_usd=0.001,
            latency_ms=10,
            estimated=False,
        )
    )

    report = collector.summary()
    assert report.warnings
    assert "exceeded threshold" in report.warnings[0]
