"""Run a single task in a fresh E2B sandbox and record the full trace."""

from __future__ import annotations

import os
import shlex
import uuid
from collections.abc import Callable
from pathlib import Path

from e2b import CommandExitException, Sandbox

from flight_recorder.models import (
    AssertionResult,
    CommandSucceeds,
    FileExists,
    OutputContains,
    RunResult,
    TaskSpec,
    TraceEvent,
)
from flight_recorder.trace import (
    LineBuffer,
    events_from_message,
    extract_metrics,
    final_result_text,
    parse_line,
)

WORKSPACE = "/home/user/workspace"
INSTALL_TIMEOUT_S = 240

OnEvent = Callable[[TraceEvent], None]


def _sandbox_exit_code(sandbox: Sandbox, command: str) -> tuple[int, str]:
    """Run a command in the workspace; non-zero exits are results, not errors."""
    try:
        result = sandbox.commands.run(command, cwd=WORKSPACE, timeout=60)
        return result.exit_code, result.stdout + result.stderr
    except Exception as exc:  # e2b raises on non-zero exit codes
        exit_code = getattr(exc, "exit_code", 1)
        output = (getattr(exc, "stdout", "") or "") + (getattr(exc, "stderr", "") or str(exc))
        return exit_code or 1, output


def _check_assertions(
    sandbox: Sandbox, task: TaskSpec, result_text: str
) -> list[AssertionResult]:
    results: list[AssertionResult] = []
    for assertion in task.assertions:
        if isinstance(assertion, OutputContains):
            passed = assertion.text in result_text
            detail = "" if passed else "text not found in final agent output"
        elif isinstance(assertion, FileExists):
            code, out = _sandbox_exit_code(sandbox, f"test -e {shlex.quote(assertion.path)}")
            passed, detail = code == 0, "" if code == 0 else f"missing: {assertion.path}"
        elif isinstance(assertion, CommandSucceeds):
            code, out = _sandbox_exit_code(sandbox, assertion.command)
            passed, detail = code == 0, "" if code == 0 else f"exit {code}: {out.strip()[:200]}"
        else:  # pragma: no cover - exhaustive over the union
            passed, detail = False, f"unknown assertion: {assertion}"
        results.append(AssertionResult(assertion=assertion, passed=passed, detail=detail))
    return results


def _ensure_claude(sandbox: Sandbox) -> None:
    sandbox.commands.run(
        "command -v claude >/dev/null || sudo npm install -g @anthropic-ai/claude-code",
        timeout=INSTALL_TIMEOUT_S,
    )


def run_task(
    task: TaskSpec,
    out_dir: Path,
    oauth_token: str | None = None,
    on_event: OnEvent | None = None,
) -> RunResult:
    """Execute `task` once in a fresh sandbox; write trace.jsonl + result.json to `out_dir`."""
    oauth_token = oauth_token or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if not oauth_token:
        raise RuntimeError(
            "CLAUDE_CODE_OAUTH_TOKEN is not set (generate one with `claude setup-token`)"
        )

    run_id = uuid.uuid4().hex[:8]
    run_dir = out_dir / task.id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.jsonl"

    events: list[TraceEvent] = []
    buffer = LineBuffer()

    sandbox = Sandbox.create(
        task.template,
        envs={"CLAUDE_CODE_OAUTH_TOKEN": oauth_token},
        timeout=task.timeout_s + INSTALL_TIMEOUT_S,
    )
    try:
        _ensure_claude(sandbox)
        sandbox.commands.run(f"mkdir -p {WORKSPACE}")

        with open(trace_path, "w", encoding="utf-8") as trace_file:

            def handle_chunk(chunk: str, *, final: bool = False) -> None:
                lines = buffer.flush() if final else buffer.feed(chunk)
                for line in lines:
                    msg = parse_line(line)
                    if msg is None:
                        continue
                    trace_file.write(line + "\n")
                    for event in events_from_message(msg, start_seq=len(events)):
                        events.append(event)
                        if on_event:
                            on_event(event)

            command = (
                f"claude -p {shlex.quote(task.prompt)} "
                "--output-format stream-json --verbose --dangerously-skip-permissions"
            )
            agent_exit_code = 0
            try:
                sandbox.commands.run(
                    command,
                    cwd=WORKSPACE,
                    timeout=task.timeout_s,
                    on_stdout=handle_chunk,
                )
            except CommandExitException as exc:  # agent failure is a result, not a crash
                agent_exit_code = exc.exit_code or 1
            finally:
                handle_chunk("", final=True)

        result_text, agent_errored = final_result_text(events)
        agent_errored = agent_errored or agent_exit_code != 0
        assertion_results = _check_assertions(sandbox, task, result_text)
        run_result = RunResult(
            task_id=task.id,
            run_id=run_id,
            sandbox_id=sandbox.sandbox_id,
            passed=not agent_errored and all(r.passed for r in assertion_results),
            agent_errored=agent_errored,
            result_text=result_text,
            assertion_results=assertion_results,
            metrics=extract_metrics(events),
            trace_path=str(trace_path),
        )
    finally:
        sandbox.kill()

    (run_dir / "result.json").write_text(run_result.model_dump_json(indent=2), encoding="utf-8")
    return run_result
