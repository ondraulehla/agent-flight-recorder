from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from flight_recorder.diff import first_divergence, render_divergence, steps_from_trace
from flight_recorder.models import RunResult, TaskSpec, TraceEvent
from flight_recorder.report import aggregate
from flight_recorder.runner import run_task

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

EVENT_STYLES = {
    "init": "dim",
    "text": "white",
    "tool_use": "cyan",
    "tool_result": "dim",
    "result": "bold",
}


@app.callback()
def main() -> None:
    """Agent flight recorder: record and evaluate agent runs in isolated sandboxes."""


def _print_event(event: TraceEvent) -> None:
    label = f"{event.tool}: " if event.tool else ""
    style = EVENT_STYLES.get(event.kind, "dim")
    console.print(f"  [{style}]{event.kind:<11} {label}{event.summary}[/{style}]")


def _print_run(result: RunResult) -> None:
    status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
    console.print(f"  {status} run_id={result.run_id}")
    for check in result.assertion_results:
        mark = "[green]✓[/green]" if check.passed else "[red]✗[/red]"
        console.print(f"    {mark} {check.assertion.kind} {check.detail}")


def _results_table(results: list[RunResult], title: str) -> Table:
    table = Table(title=title)
    for col in ("run", "status", "turns", "tool calls", "duration", "cost"):
        table.add_column(col)
    for r in results:
        m = r.metrics
        table.add_row(
            r.run_id,
            "PASS" if r.passed else "FAIL",
            str(m.num_turns or "?"),
            str(m.tool_calls),
            f"{m.duration_ms / 1000:.1f}s" if m.duration_ms else "?",
            f"${m.total_cost_usd:.4f}" if m.total_cost_usd is not None else "?",
        )
    return table


def _print_summary(results: list[RunResult]) -> None:
    agg = aggregate(results)
    console.print()
    console.print(_results_table(results, f"{agg.task_id}: {agg.passes}/{agg.runs} passed"))

    parts = [f"pass rate [bold]{agg.pass_rate:.0%}[/bold]"]
    if agg.flaky:
        parts.append("[yellow bold]FLAKY[/yellow bold]")
    if agg.mean_duration_ms is not None:
        duration = f"duration {agg.mean_duration_ms / 1000:.1f}s"
        if agg.stdev_duration_ms is not None:
            duration += f" ±{agg.stdev_duration_ms / 1000:.1f}s"
        parts.append(duration)
    if agg.cost_per_success_usd is not None:
        parts.append(f"cost per success ${agg.cost_per_success_usd:.4f}")
    console.print("  " + "  ·  ".join(parts))


def _print_pass_fail_divergence(results: list[RunResult]) -> None:
    """When runs disagree, show where a passing and a failing trajectory split."""
    passing = next((r for r in results if r.passed), None)
    failing = next((r for r in results if not r.passed), None)
    if not passing or not failing:
        return
    divergence = first_divergence(
        steps_from_trace(passing.trace_path), steps_from_trace(failing.trace_path)
    )
    console.print(f"\n[yellow]PASS {passing.run_id} vs FAIL {failing.run_id}:[/yellow]")
    for line in render_divergence(divergence, f"PASS {passing.run_id}", f"FAIL {failing.run_id}"):
        console.print(f"  {line}")


@app.command()
def run(
    task_file: Annotated[Path, typer.Argument(help="Path to a task YAML file")],
    runs: Annotated[int, typer.Option("--runs", "-n", min=1, help="How many runs")] = 1,
    jobs: Annotated[int, typer.Option("--jobs", "-j", min=1, help="Parallel sandboxes")] = 4,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory")] = Path("runs"),
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Do not stream events")] = False,
) -> None:
    """Run a task in fresh E2B sandbox(es) and record traces."""
    load_dotenv()
    task = TaskSpec.from_yaml(task_file)
    stream_events = runs == 1 and not quiet
    results: list[RunResult] = []
    crashes = 0

    console.print(f"[bold]{task.id}[/bold]: {runs} run(s), up to {min(jobs, runs)} in parallel")
    with ThreadPoolExecutor(max_workers=min(jobs, runs)) as pool:
        on_event = _print_event if stream_events else None
        futures = [
            pool.submit(run_task, task, out_dir=out, on_event=on_event) for _ in range(runs)
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - a crashed run must not kill the batch
                crashes += 1
                console.print(f"  [red]CRASH[/red] {exc}")
                continue
            results.append(result)
            _print_run(result)

    if crashes:
        console.print(f"[red]{crashes}/{runs} runs crashed (harness/sandbox error)[/red]")
    if results:
        _print_summary(results)
        _print_pass_fail_divergence(results)

    all_passed = not crashes and all(r.passed for r in results)
    raise typer.Exit(code=0 if all_passed else 1)


@app.command()
def diff(
    trace_a: Annotated[Path, typer.Argument(help="First trace.jsonl")],
    trace_b: Annotated[Path, typer.Argument(help="Second trace.jsonl")],
) -> None:
    """Show where two recorded trajectories diverged (first differing tool call)."""
    left, right = steps_from_trace(trace_a), steps_from_trace(trace_b)
    console.print(f"A: {trace_a} ({len(left)} steps)")
    console.print(f"B: {trace_b} ({len(right)} steps)")
    for line in render_divergence(first_divergence(left, right), "A", "B"):
        console.print(line)


if __name__ == "__main__":
    app()
