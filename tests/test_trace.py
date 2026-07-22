import json

from flight_recorder.trace import (
    LineBuffer,
    events_from_message,
    extract_metrics,
    final_result_text,
    parse_line,
)

INIT = {"type": "system", "subtype": "init", "model": "claude-sonnet-5", "tools": ["Bash"]}
ASSISTANT = {
    "type": "assistant",
    "message": {
        "content": [
            {"type": "text", "text": "I'll create the file now."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "echo hi > hello.txt"}},
        ]
    },
}
TOOL_RESULT = {
    "type": "user",
    "message": {"content": [{"type": "tool_result", "content": "done"}]},
}
RESULT = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "Created hello.txt as requested.",
    "duration_ms": 8500,
    "num_turns": 4,
    "total_cost_usd": 0.0421,
    "usage": {"input_tokens": 1200, "output_tokens": 340},
}


def _events(*messages):
    events = []
    for msg in messages:
        events.extend(events_from_message(msg, start_seq=len(events)))
    return events


class TestLineBuffer:
    def test_reassembles_chunks_split_mid_line(self):
        buf = LineBuffer()
        assert buf.feed('{"a"') == []
        assert buf.feed(': 1}\n{"b": 2}\n{"c"') == ['{"a": 1}', '{"b": 2}']
        assert buf.flush() == ['{"c"']

    def test_skips_blank_lines(self):
        buf = LineBuffer()
        assert buf.feed("\n\n  \nx\n") == ["x"]
        assert buf.flush() == []


class TestParseLine:
    def test_valid_json_object(self):
        assert parse_line(json.dumps(INIT)) == INIT

    def test_noise_returns_none(self):
        assert parse_line("not json") is None
        assert parse_line('"a bare string"') is None


class TestEventsFromMessage:
    def test_assistant_message_expands_per_block(self):
        events = events_from_message(ASSISTANT, start_seq=5)
        assert [e.kind for e in events] == ["text", "tool_use"]
        assert [e.seq for e in events] == [5, 6]
        assert events[1].tool == "Bash"
        assert "hello.txt" in events[1].summary

    def test_full_session(self):
        events = _events(INIT, ASSISTANT, TOOL_RESULT, RESULT)
        assert [e.kind for e in events] == ["init", "text", "tool_use", "tool_result", "result"]
        assert [e.seq for e in events] == list(range(5))

    def test_long_summaries_are_truncated(self):
        msg = {"type": "assistant", "message": {"content": [{"type": "text", "text": "x" * 999}]}}
        (event,) = events_from_message(msg, start_seq=0)
        assert len(event.summary) == 200

    def test_tool_result_with_block_list_content(self):
        msg = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "content": [{"type": "text", "text": "output here"}]}
                ]
            },
        }
        (event,) = events_from_message(msg, start_seq=0)
        assert event.summary == "output here"


class TestMetricsAndResult:
    def test_extracts_metrics_from_result_event(self):
        metrics = extract_metrics(_events(INIT, ASSISTANT, TOOL_RESULT, RESULT))
        assert metrics.tool_calls == 1
        assert metrics.num_turns == 4
        assert metrics.total_cost_usd == 0.0421
        assert metrics.input_tokens == 1200

    def test_final_text_success(self):
        text, errored = final_result_text(_events(INIT, RESULT))
        assert text == "Created hello.txt as requested."
        assert errored is False

    def test_missing_result_means_errored(self):
        text, errored = final_result_text(_events(INIT, ASSISTANT))
        assert text == ""
        assert errored is True

    def test_error_subtype_means_errored(self):
        error_result = {**RESULT, "subtype": "error_max_turns"}
        _, errored = final_result_text(_events(error_result))
        assert errored is True


def _rate_limit(status: str) -> dict:
    return {"type": "rate_limit_event", "rate_limit_info": {"status": status, "resetsAt": 1}}


class TestRateLimit:
    def test_event_gets_readable_summary(self):
        (event,) = events_from_message(_rate_limit("allowed_warning"), start_seq=0)
        assert event.kind == "other"
        assert event.summary == "rate limit: allowed_warning"

    def test_no_events_means_none(self):
        assert extract_metrics(_events(INIT, RESULT)).rate_limit_status is None

    def test_worst_status_wins(self):
        events = _events(_rate_limit("allowed"), _rate_limit("rejected"), _rate_limit("allowed"))
        assert extract_metrics(events).rate_limit_status == "rejected"

    def test_all_allowed(self):
        events = _events(_rate_limit("allowed"), _rate_limit("allowed"))
        assert extract_metrics(events).rate_limit_status == "allowed"
