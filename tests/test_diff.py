import json

from flight_recorder.diff import (
    Step,
    first_divergence,
    render_divergence,
    steps_from_trace,
)


def _assistant_line(*blocks) -> str:
    return json.dumps({"type": "assistant", "message": {"content": list(blocks)}})


def _tool(name: str, **input_args) -> dict:
    return {"type": "tool_use", "name": name, "input": input_args}


def _write_trace(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return path


def test_steps_from_trace_extracts_actions_only(tmp_path):
    trace = _write_trace(
        tmp_path / "trace.jsonl",
        [
            json.dumps({"type": "system", "subtype": "init", "model": "m"}),
            _assistant_line(
                {"type": "text", "text": "thinking..."},
                _tool("Write", file_path="a.txt", content="x"),
                _tool("Write", file_path="b.txt", content="y"),
            ),
            json.dumps({"type": "result", "subtype": "success", "result": "done"}),
            "not json at all",
        ],
    )
    steps = steps_from_trace(trace)
    assert [s.tool for s in steps] == ["Write", "Write"]
    assert "a.txt" in steps[0].detail
    assert "b.txt" in steps[1].detail  # same tool twice keeps per-block inputs


def test_canonical_input_ordering(tmp_path):
    trace_a = _write_trace(
        tmp_path / "a.jsonl", [_assistant_line({"type": "tool_use", "name": "Bash",
                                                "input": {"command": "ls", "timeout": 5}})]
    )
    trace_b = _write_trace(
        tmp_path / "b.jsonl", [_assistant_line({"type": "tool_use", "name": "Bash",
                                                "input": {"timeout": 5, "command": "ls"}})]
    )
    assert steps_from_trace(trace_a) == steps_from_trace(trace_b)


def _steps(*pairs) -> list[Step]:
    return [Step(tool=t, detail=d) for t, d in pairs]


def test_identical_trajectories():
    steps = _steps(("Write", "{}"), ("Bash", "{}"))
    div = first_divergence(steps, list(steps))
    assert div.index is None
    assert div.common == 2
    assert render_divergence(div, "A", "B") == ["identical trajectories (2 steps)"]


def test_divergence_in_the_middle():
    left = _steps(("Write", "{}"), ("Bash", '{"command": "pytest"}'))
    right = _steps(("Write", "{}"), ("Edit", "{}"))
    div = first_divergence(left, right)
    assert div.index == 1
    assert div.common == 1
    assert div.left is not None and div.left.tool == "Bash"
    assert div.right is not None and div.right.tool == "Edit"


def test_one_trajectory_is_prefix_of_other():
    left = _steps(("Write", "{}"))
    right = _steps(("Write", "{}"), ("Bash", "{}"))
    div = first_divergence(left, right)
    assert div.index == 1
    assert div.left is None
    assert div.right is not None
    assert "(no more steps)" in render_divergence(div, "A", "B")[1]
