import pytest

from flight_recorder.models import RunMetrics, RunResult
from flight_recorder.report import aggregate


def _result(
    passed: bool,
    duration_ms: int | None = 10_000,
    cost: float | None = 0.05,
    tool_calls: int = 2,
) -> RunResult:
    return RunResult(
        task_id="t",
        run_id="r",
        passed=passed,
        trace_path="trace.jsonl",
        metrics=RunMetrics(
            duration_ms=duration_ms, total_cost_usd=cost, tool_calls=tool_calls
        ),
    )


def test_all_passed():
    agg = aggregate([_result(True), _result(True)])
    assert agg.pass_rate == 1.0
    assert agg.flaky is False
    assert agg.cost_per_success_usd == pytest.approx(0.05)
    assert agg.total_cost_usd == pytest.approx(0.10)


def test_mixed_results_are_flaky():
    agg = aggregate([_result(True), _result(False), _result(False)])
    assert agg.pass_rate == pytest.approx(1 / 3)
    assert agg.flaky is True
    # total cost is spread over successes only
    assert agg.cost_per_success_usd == pytest.approx(0.15)


def test_no_passes_has_no_cost_per_success():
    agg = aggregate([_result(False)])
    assert agg.pass_rate == 0.0
    assert agg.flaky is False
    assert agg.cost_per_success_usd is None


def test_missing_metrics_tolerated():
    agg = aggregate([_result(True, duration_ms=None, cost=None)])
    assert agg.mean_duration_ms is None
    assert agg.stdev_duration_ms is None
    assert agg.total_cost_usd is None


def test_stdev_needs_two_durations():
    assert aggregate([_result(True)]).stdev_duration_ms is None
    agg = aggregate([_result(True, duration_ms=8000), _result(True, duration_ms=12000)])
    assert agg.mean_duration_ms == pytest.approx(10_000)
    assert agg.stdev_duration_ms is not None


def test_empty_results_rejected():
    with pytest.raises(ValueError):
        aggregate([])
