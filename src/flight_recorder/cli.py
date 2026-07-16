from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from flight_recorder.models import TaskSpec, TraceEvent
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


def _print_event(event: TraceEvent) -> None:
    label = f"{event.tool}: " if event.tool else ""
    style = EVENT_STYLES.get(event.kind, "dim")
    console.print(f"  [{style}]{event.kind:<11} {label}{event.summary}[/{style}]")


@app.callback()
def main() -> None:
    """Agent flight recorder: record and evaluate agent runs in isolated sandboxes."""


@app.command()
def run(
    task_file: Annotated[Path, typer.Argument(help="Path to a task YAML file")],
    runs: Annotated[int, typer.Option("--runs", "-n", min=1, help="How many runs")] = 1,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory")] = Path("runs"),
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Do not stream events")] = False,
) -> None:
    """Run a task in fresh E2B sandbox(es) and record traces."""
    load_dotenv()
    task = TaskSpec.from_yaml(task_file)
    results = []

    for i in range(runs):
        console.print(f"\n[bold]run {i + 1}/{runs}[/bold] task={task.id}")
        result = run_task(task, out_dir=out, on_event=None if quiet else _print_event)
        results.append(result)
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(f"  {status} run_id={result.run_id}")
        for check in result.assertion_results:
            mark = "[green]✓[/green]" if check.passed else "[red]✗[/red]"
            console.print(f"    {mark} {check.assertion.kind} {check.detail}")

    table = Table(title=f"{task.id}: {sum(r.passed for r in results)}/{runs} passed")
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
    console.print()
    console.print(table)

    raise typer.Exit(code=0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    app()
