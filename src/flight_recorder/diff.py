"""Compare two recorded runs: where did the trajectories diverge?

A trajectory is the sequence of *actions* (tool calls) an agent took. Free-text
wording differs between runs even when behaviour is identical, so text events
are excluded. Two refinements over a naive comparison:

- fuzzy step matching: `head -60` vs `head -80` is the same action, not a
  divergence, so steps match when the tool is identical and the inputs are
  similar enough (SequenceMatcher ratio >= threshold);
- LCS alignment instead of prefix comparison: one extra exploratory step in
  the middle must not mark everything after it as diverged.

Pure module: works on trace.jsonl files, no sandbox.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from flight_recorder.trace import parse_line

DEFAULT_THRESHOLD = 0.7
RENDER_MAX_CHARS = 120


class Step(BaseModel):
    tool: str
    detail: str

    def render(self, max_chars: int = RENDER_MAX_CHARS) -> str:
        text = f"{self.tool}({self.detail})"
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


class AlignedStep(BaseModel):
    op: Literal["match", "left_only", "right_only"]
    left: Step | None = None
    right: Step | None = None
    similarity: float | None = None


class Divergence(BaseModel):
    index: int | None  # position in the alignment; None = fully aligned
    common: int  # matched steps before the divergence (or total when aligned)
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


def step_similarity(a: Step, b: Step) -> float:
    if a.tool != b.tool:
        return 0.0
    return SequenceMatcher(None, a.detail, b.detail).ratio()


def align(
    left: list[Step], right: list[Step], threshold: float = DEFAULT_THRESHOLD
) -> list[AlignedStep]:
    """Weighted LCS: pair up similar steps, maximizing total similarity of matches."""
    n, m = len(left), len(right)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            similarity = step_similarity(left[i], right[j])
            best = max(score[i + 1][j], score[i][j + 1])
            if similarity >= threshold:
                best = max(best, similarity + score[i + 1][j + 1])
            score[i][j] = best

    ops: list[AlignedStep] = []
    i = j = 0
    while i < n and j < m:
        similarity = step_similarity(left[i], right[j])
        if similarity >= threshold and score[i][j] == similarity + score[i + 1][j + 1]:
            ops.append(
                AlignedStep(op="match", left=left[i], right=right[j], similarity=similarity)
            )
            i, j = i + 1, j + 1
        elif score[i + 1][j] >= score[i][j + 1]:
            ops.append(AlignedStep(op="left_only", left=left[i]))
            i += 1
        else:
            ops.append(AlignedStep(op="right_only", right=right[j]))
            j += 1
    ops.extend(AlignedStep(op="left_only", left=step) for step in left[i:])
    ops.extend(AlignedStep(op="right_only", right=step) for step in right[j:])
    return ops


def first_divergence(
    left: list[Step], right: list[Step], threshold: float = DEFAULT_THRESHOLD
) -> Divergence:
    ops = align(left, right, threshold)
    matched_prefix = 0
    for op in ops:
        if op.op != "match":
            break
        matched_prefix += 1
    else:
        return Divergence(index=None, common=len(ops))
    rest = ops[matched_prefix:]
    return Divergence(
        index=matched_prefix,
        common=matched_prefix,
        left=next((o.left for o in rest if o.op == "left_only"), None),
        right=next((o.right for o in rest if o.op == "right_only"), None),
    )


def render_divergence(div: Divergence, left_name: str, right_name: str) -> list[str]:
    if div.index is None:
        return [f"identical trajectories ({div.common} steps)"]
    lines = [f"diverged at step {div.index} (after {div.common} common steps)"]
    for name, step in ((left_name, div.left), (right_name, div.right)):
        action = step.render() if step else "(no more steps)"
        lines.append(f"  {name}: {action}")
    return lines


def render_alignment(ops: list[AlignedStep]) -> list[str]:
    """Unified-diff-style listing: '=' exact, '≈' fuzzy match, '-'/'+' one-sided."""
    lines = []
    for op in ops:
        if op.op == "match":
            exact = op.similarity is not None and op.similarity > 0.999
            symbol = "=" if exact else "≈"
            step = op.left.render() if op.left else ""
            suffix = "" if exact or op.similarity is None else f"  [{op.similarity:.0%} similar]"
            lines.append(f" {symbol} {step}{suffix}")
        elif op.op == "left_only":
            lines.append(f" - {op.left.render() if op.left else ''}")
        else:
            lines.append(f" + {op.right.render() if op.right else ''}")
    return lines
