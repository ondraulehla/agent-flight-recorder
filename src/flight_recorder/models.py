from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field


class FileExists(BaseModel):
    """Passes when `path` (relative to the agent workspace) exists in the sandbox."""

    kind: Literal["file_exists"] = "file_exists"
    path: str


class OutputContains(BaseModel):
    """Passes when the agent's final result text contains `text`."""

    kind: Literal["output_contains"] = "output_contains"
    text: str


class CommandSucceeds(BaseModel):
    """Passes when `command` exits 0 inside the sandbox workspace."""

    kind: Literal["command_succeeds"] = "command_succeeds"
    command: str


Assertion = Annotated[
    FileExists | OutputContains | CommandSucceeds,
    Field(discriminator="kind"),
]


class SetupFile(BaseModel):
    """A file written into the workspace before the agent starts."""

    path: str
    content: str


class TaskSpec(BaseModel):
    id: str
    prompt: str
    assertions: list[Assertion] = []
    setup_files: list[SetupFile] = []
    setup_commands: list[str] = []
    template: str = "base"
    timeout_s: int = 600

    @classmethod
    def from_yaml(cls, path: Path | str) -> TaskSpec:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


class TraceEvent(BaseModel):
    seq: int
    kind: Literal["init", "text", "tool_use", "tool_result", "result", "other"]
    tool: str | None = None
    summary: str = ""
    raw: dict


class RunMetrics(BaseModel):
    duration_ms: int | None = None
    num_turns: int | None = None
    total_cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: int = 0


class AssertionResult(BaseModel):
    assertion: Assertion
    passed: bool
    detail: str = ""


class RunResult(BaseModel):
    task_id: str
    run_id: str
    sandbox_id: str | None = None
    model: str | None = None
    passed: bool
    agent_errored: bool = False
    result_text: str = ""
    assertion_results: list[AssertionResult] = []
    metrics: RunMetrics = RunMetrics()
    trace_path: str
