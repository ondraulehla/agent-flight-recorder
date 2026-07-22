from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from flight_recorder.diff import (
    DEFAULT_THRESHOLD,
    align,
    first_divergence,
    render_alignment,
    render_divergence,
    steps_from_trace,
)
from flight_recorder.models import RunResult, TaskSpec, TraceEvent
from flight_recorder.report import aggregate, load_results
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


def _run_batch(
    task: TaskSpec,
    out: Path,
    runs: int,
    jobs: int,
    stream_events: bool,
    model: str | None = None,
) -> tuple[list[RunResult], int]:
    """Run a task N times in parallel sandboxes; returns (results, crash count)."""
    results: list[RunResult] = []
    crashes = 0
    label = f" model={model}" if model else ""
    console.print(
        f"[bold]{task.id}[/bold]:{label} {runs} run(s), up to {min(jobs, runs)} in parallel"
    )
    with ThreadPoolExecutor(max_workers=min(jobs, runs)) as pool:
        on_event = _print_event if stream_events else None
        futures = [
            pool.submit(run_task, task, out_dir=out, on_event=on_event, model=model)
            for _ in range(runs)
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
    return results, crashes


@app.command()
def run(
    task_file: Annotated[Path, typer.Argument(help="Path to a task YAML file")],
    runs: Annotated[int, typer.Option("--runs", "-n", min=1, help="How many runs")] = 1,
    jobs: Annotated[int, typer.Option("--jobs", "-j", min=1, help="Parallel sandboxes")] = 4,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory")] = Path("runs"),
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model for the agent (claude --model)")
    ] = "",
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Do not stream events")] = False,
) -> None:
    """Run a task in fresh E2B sandbox(es) and record traces."""
    load_dotenv()
    task = TaskSpec.from_yaml(task_file)
    results, crashes = _run_batch(
        task, out, runs, jobs, stream_events=runs == 1 and not quiet, model=model or None
    )
    if results:
        _print_summary(results)
        _print_pass_fail_divergence(results)
    all_passed = not crashes and all(r.passed for r in results)
    raise typer.Exit(code=0 if all_passed else 1)


@app.command()
def compare(
    task_file: Annotated[Path, typer.Argument(help="Path to a task YAML file")],
    models: Annotated[
        list[str],
        typer.Option("--model", "-m", help="Model to compare (repeat for each)"),
    ],
    runs: Annotated[int, typer.Option("--runs", "-n", min=1, help="Runs per model")] = 3,
    jobs: Annotated[int, typer.Option("--jobs", "-j", min=1, help="Parallel sandboxes")] = 4,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory")] = Path("runs"),
) -> None:
    """Run the same task under multiple models and compare reliability and cost."""
    load_dotenv()
    task = TaskSpec.from_yaml(task_file)
    by_model: dict[str, list[RunResult]] = {}
    for model in models:
        results, _ = _run_batch(task, out, runs, jobs, stream_events=False, model=model)
        by_model[model] = results

    table = Table(title=f"{task.id}: {runs} runs per model")
    for col in ("model", "pass rate", "flaky", "duration", "tool calls", "cost per success"):
        table.add_column(col)
    for model, results in by_model.items():
        if not results:
            table.add_row(model, "-", "-", "-", "-", "-")
            continue
        agg = aggregate(results)
        duration = f"{agg.mean_duration_ms / 1000:.1f}s" if agg.mean_duration_ms else "?"
        if agg.stdev_duration_ms is not None:
            duration += f" ±{agg.stdev_duration_ms / 1000:.1f}s"
        cost = f"${agg.cost_per_success_usd:.4f}" if agg.cost_per_success_usd else "-"
        table.add_row(
            model,
            f"{agg.passes}/{agg.runs} ({agg.pass_rate:.0%})",
            "yes" if agg.flaky else "no",
            duration,
            f"{agg.mean_tool_calls:.1f}" if agg.mean_tool_calls is not None else "?",
            cost,
        )
    console.print()
    console.print(table)


@app.command()
def report(
    task_id: Annotated[str, typer.Argument(help="Task id (directory name under runs/)")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Runs directory")] = Path("runs"),
) -> None:
    """Aggregate all recorded runs of a task, across sessions."""
    results = load_results(out, task_id)
    if not results:
        console.print(f"[red]no recorded runs for '{task_id}' in {out}/[/red]")
        raise typer.Exit(code=1)
    _print_summary(results)
    _print_pass_fail_divergence(results)


@app.command()
def build_template(
    cpu_count: Annotated[int, typer.Option(min=1)] = 1,
    memory_mb: Annotated[int, typer.Option(min=512)] = 1024,
) -> None:
    """Build the E2B template with the agent preinstalled (fast cold starts)."""
    load_dotenv()
    from flight_recorder.template import build_template as build

    name = build(cpu_count=cpu_count, memory_mb=memory_mb)
    console.print(f"[green]template '{name}' built[/green] – use `template: {name}` in tasks")


@app.command()
def diff(
    trace_a: Annotated[Path, typer.Argument(help="First trace.jsonl")],
    trace_b: Annotated[Path, typer.Argument(help="Second trace.jsonl")],
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", min=0.0, max=1.0,
                     help="Similarity needed for two steps to count as the same action"),
    ] = DEFAULT_THRESHOLD,
) -> None:
    """Align two recorded trajectories and show where they truly diverged."""
    left, right = steps_from_trace(trace_a), steps_from_trace(trace_b)
    console.print(f"A (-): {trace_a} ({len(left)} steps)")
    console.print(f"B (+): {trace_b} ({len(right)} steps)")
    ops = align(left, right, threshold)
    for line in render_alignment(ops):
        style = {"-": "red", "+": "green", "≈": "yellow"}.get(line.lstrip()[:1], "dim")
        console.print(f"[{style}]{line}[/{style}]", highlight=False)
    matched = sum(1 for op in ops if op.op == "match")
    console.print(f"\n{matched} matched · {len(left) - matched} only in A "
                  f"· {len(right) - matched} only in B")


if __name__ == "__main__":
    app()
