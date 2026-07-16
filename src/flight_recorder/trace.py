"""Parsing of Claude Code `--output-format stream-json` output into trace events.

Everything here is pure (no network, no sandbox) so it can be unit-tested
against captured stream lines.
"""

from __future__ import annotations

import json
from typing import Any

from flight_recorder.models import RunMetrics, TraceEvent

SUMMARY_MAX_CHARS = 200


class LineBuffer:
    """Reassembles complete lines from arbitrarily chunked stdout callbacks."""

    def __init__(self) -> None:
        self._partial = ""

    def feed(self, chunk: str) -> list[str]:
        self._partial += chunk
        *complete, self._partial = self._partial.split("\n")
        return [line for line in complete if line.strip()]

    def flush(self) -> list[str]:
        rest, self._partial = self._partial, ""
        return [rest] if rest.strip() else []


def parse_line(line: str) -> dict[str, Any] | None:
    """Parse one stream line into a raw message dict; None for non-JSON noise."""
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return None
    return msg if isinstance(msg, dict) else None


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > SUMMARY_MAX_CHARS:
        return text[: SUMMARY_MAX_CHARS - 1] + "…"
    return text


def _content_blocks(msg: dict[str, Any]) -> list[dict[str, Any]]:
    content = msg.get("message", {}).get("content", [])
    return content if isinstance(content, list) else []


def events_from_message(msg: dict[str, Any], start_seq: int) -> list[TraceEvent]:
    """Expand one stream-json message into zero or more trace events."""
    events: list[TraceEvent] = []

    def add(kind: str, summary: str, tool: str | None = None) -> None:
        events.append(
            TraceEvent(
                seq=start_seq + len(events),
                kind=kind,  # type: ignore[arg-type]
                tool=tool,
                summary=_truncate(summary),
                raw=msg,
            )
        )

    msg_type = msg.get("type")
    if msg_type == "system" and msg.get("subtype") == "init":
        add("init", f"session started (model={msg.get('model', '?')})")
    elif msg_type == "assistant":
        for block in _content_blocks(msg):
            if block.get("type") == "text" and block.get("text"):
                add("text", block["text"])
            elif block.get("type") == "tool_use":
                add("tool_use", json.dumps(block.get("input", {})), tool=block.get("name"))
    elif msg_type == "user":
        for block in _content_blocks(msg):
            if block.get("type") == "tool_result":
                content = block.get("content")
                if isinstance(content, list):
                    content = " ".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                add("tool_result", str(content or ""))
    elif msg_type == "result":
        add("result", f"{msg.get('subtype', 'unknown')}: {msg.get('result', '')}")
    else:
        add("other", str(msg_type))
    return events


def extract_metrics(events: list[TraceEvent]) -> RunMetrics:
    """Pull run metrics from the final `result` message, plus tool-call count."""
    metrics = RunMetrics(tool_calls=sum(1 for e in events if e.kind == "tool_use"))
    result = next((e.raw for e in reversed(events) if e.kind == "result"), None)
    if result is None:
        return metrics
    usage = result.get("usage") or {}
    metrics.duration_ms = result.get("duration_ms")
    metrics.num_turns = result.get("num_turns")
    metrics.total_cost_usd = result.get("total_cost_usd")
    metrics.input_tokens = usage.get("input_tokens")
    metrics.output_tokens = usage.get("output_tokens")
    return metrics


def final_result_text(events: list[TraceEvent]) -> tuple[str, bool]:
    """Return (agent's final text, errored?) from the trailing result event."""
    result = next((e.raw for e in reversed(events) if e.kind == "result"), None)
    if result is None:
        return "", True
    errored = result.get("is_error", False) or result.get("subtype") != "success"
    return str(result.get("result") or ""), bool(errored)
