# agent-flight-recorder

A flight recorder for AI agents: run an agent task N times in isolated [E2B](https://e2b.dev) sandboxes, record the full trace of every run (each tool call, tokens, cost), and measure what actually matters in production – pass rate, flakiness and cost per success.

Agents are non-deterministic: the same prompt can take a different execution path every run. A demo that works once tells you nothing. This tool treats every agent run like a flight: it happens in a clean, isolated sandbox, and everything is recorded so you can find out where two flights diverged.

## Status

Current: tasks with assertions, full trace to JSONL, parallel N runs with a pass-rate / flakiness / cost-per-success report, and trajectory diff (first tool call where two runs diverged). When a batch has both passing and failing runs, the divergence between them is shown automatically.

Tasks can ship setup fixtures (files and commands prepared in the workspace before the agent starts), so you can record realistic scenarios like "fix the bug so the tests pass" – see [tasks/fix-median-bug.yaml](tasks/fix-median-bug.yaml).

Roadmap:
- [ ] Record rate-limit events into run metrics
- [ ] Compare pass rate / cost across models

## How it works

1. You define a task in YAML: a prompt plus assertions (what "success" means).
2. The runner boots a fresh E2B sandbox and executes the task with Claude Code in headless mode (`claude -p --output-format stream-json`).
3. Every streamed message (assistant text, tool calls, tool results, final result) is parsed into trace events and appended to `runs/<task>/<run-id>/trace.jsonl`.
4. Assertions are evaluated inside the sandbox (file checks, arbitrary commands) and against the agent's final answer.
5. You get a `result.json` with pass/fail, per-assertion detail, turns, tool calls, duration and cost.

No Anthropic API key is needed – the agent authenticates with a Claude subscription OAuth token.

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in `.env`:
- `E2B_API_KEY` – from the [E2B dashboard](https://e2b.dev/dashboard)
- `CLAUDE_CODE_OAUTH_TOKEN` – run `claude setup-token` on a machine with Claude Code logged in

## Usage

```bash
uv run flightrec run tasks/hello-world.yaml        # one recorded run, events streamed live
uv run flightrec run tasks/hello-world.yaml -n 5   # five parallel runs, flakiness summary
uv run flightrec run tasks/hello-world.yaml -n 8 -j 2   # limit sandbox concurrency

# build the E2B template with the agent preinstalled (once; fast cold starts)
uv run flightrec build-template

# aggregate every recorded run of a task, across sessions
uv run flightrec report hello-world

# where did two runs take a different path?
uv run flightrec diff runs/hello-world/<run-a>/trace.jsonl runs/hello-world/<run-b>/trace.jsonl
```

Tasks reference the prebuilt template via `template: flight-recorder`; set `template: base` to run without building one (the runner then installs the agent at sandbox startup, which is slower).

Example task:

```yaml
id: hello-world
prompt: |
  Create a file named hello.txt containing exactly the line "Hello from the sandbox!"
assertions:
  - kind: file_exists
    path: hello.txt
  - kind: command_succeeds
    command: grep -qx "Hello from the sandbox!" hello.txt
timeout_s: 300
```

Assertion kinds: `file_exists` (path in workspace), `command_succeeds` (exit 0 in sandbox), `output_contains` (substring of the agent's final answer).

## Development

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

The trace-parsing layer (`trace.py`) is pure and fully unit-tested; sandbox I/O lives in `runner.py`.
