"""Aggregate metrics across N runs of the same task. Pure, unit-testable."""

from __future__ import annotations

import statistics
from pathlib import Path

from pydantic import BaseModel

from flight_recorder.models import RunResult


def load_results(runs_dir: Path, task_id: str) -> list[RunResult]:
    """Load all persisted result.json files for a task, oldest first."""
    result_files = sorted(
        (runs_dir / task_id).glob("*/result.json"), key=lambda p: p.stat().st_mtime
    )
    return [RunResult.model_validate_json(p.read_text(encoding="utf-8")) for p in result_files]


class Aggregate(BaseModel):
    task_id: str
    runs: int
    passes: int
    pass_rate: float
    flaky: bool
    mean_duration_ms: float | None = None
    stdev_duration_ms: float | None = None
    mean_tool_calls: float | None = None
    total_cost_usd: float | None = None
    cost_per_success_usd: float | None = None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def aggregate(results: list[RunResult]) -> Aggregate:
    if not results:
        raise ValueError("no results to aggregate")

    passes = sum(1 for r in results if r.passed)
    durations = [float(r.metrics.duration_ms) for r in results if r.metrics.duration_ms]
    costs = [r.metrics.total_cost_usd for r in results if r.metrics.total_cost_usd is not None]
    total_cost = sum(costs) if costs else None

    return Aggregate(
        task_id=results[0].task_id,
        runs=len(results),
        passes=passes,
        pass_rate=passes / len(results),
        flaky=0 < passes < len(results),
        mean_duration_ms=_mean(durations),
        stdev_duration_ms=statistics.stdev(durations) if len(durations) >= 2 else None,
        mean_tool_calls=_mean([float(r.metrics.tool_calls) for r in results]),
        total_cost_usd=total_cost,
        cost_per_success_usd=total_cost / passes if total_cost is not None and passes else None,
    )
