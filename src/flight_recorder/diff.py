"""Compare two recorded runs: where did the trajectories diverge?

A trajectory is the sequence of *actions* (tool calls) an agent took. Free-text
wording differs between runs even when behaviour is identical, so text events
are excluded by default. Pure module: works on trace.jsonl files, no sandbox.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from flight_recorder.trace import parse_line


class Step(BaseModel):
    tool: str
    detail: str

    def render(self) -> str:
        return f"{self.tool}({self.detail})"


class Divergence(BaseModel):
    index: int | None  # None = identical trajectories
    common: int
    left: Step | None = None
    right: Step | None = None


def steps_from_trace(path: Path | str) -> list[Step]:
    """Extract the action sequence (tool calls with canonical inputs) from a trace file."""
    steps: list[Step] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            msg = parse_line(line)
            if msg is None or msg.get("type") != "assistant":
                continue
            content = msg.get("message", {}).get("content", [])
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "tool_use":
                    detail = json.dumps(block.get("input", {}), sort_keys=True)
                    steps.append(Step(tool=block.get("name", "?"), detail=detail))
    return steps


def first_divergence(left: list[Step], right: list[Step]) -> Divergence:
    common = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        common += 1
    if common == len(left) == len(right):
        return Divergence(index=None, common=common)
    return Divergence(
        index=common,
        common=common,
        left=left[common] if common < len(left) else None,
        right=right[common] if common < len(right) else None,
    )


def render_divergence(div: Divergence, left_name: str, right_name: str) -> list[str]:
    if div.index is None:
        return [f"identical trajectories ({div.common} steps)"]
    lines = [f"diverged at step {div.index} (after {div.common} common steps)"]
    for name, step in ((left_name, div.left), (right_name, div.right)):
        action = step.render() if step else "(no more steps)"
        lines.append(f"  {name}: {action}")
    return lines
