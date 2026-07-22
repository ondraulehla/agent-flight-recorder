# agent-flight-recorder

A flight recorder for AI agents: run an agent task N times in isolated [E2B](https://e2b.dev) sandboxes, record the full trace of every run (each tool call, tokens, cost), and measure what actually matters in production – pass rate, flakiness and cost per success.

Agents are non-deterministic: the same prompt can take a different execution path every run. A demo that works once tells you nothing. This tool treats every agent run like a flight: it happens in a clean, isolated sandbox, and everything is recorded so you can find out where two flights diverged.

## Status

Current: tasks with assertions, full trace to JSONL, parallel N runs with a pass-rate / flakiness / cost-per-success report, and trajectory diff (first tool call where two runs diverged). When a batch has both passing and failing runs, the divergence between them is shown automatically.

Tasks can ship setup fixtures (files and commands prepared in the workspace before the agent starts), so you can record realistic scenarios like "fix the bug so the tests pass" – see [tasks/fix-median-bug.yaml](tasks/fix-median-bug.yaml).

Rate-limit events from the stream are recorded too: each run stores the most severe status seen (`allowed` / `allowed_warning` / `rejected`), and the batch summary warns when any run was throttled – so a slow batch is distinguishable from a throttled one.

Roadmap:
- [ ] HTML report with embedded trace viewer

## Demo

`tasks/fix-median-bug.yaml` plants two bugs in a small `stats.py` (a crash on empty input and an off-by-one in the median) plus a test suite that catches them. The agent is asked to make the tests pass; a checksum assertion guards against the agent "fixing" the tests instead. Three parallel recorded runs:

```
$ flightrec run tasks/fix-median-bug.yaml -n 3

                  fix-median-bug: 3/3 passed
┏━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ run      ┃ status ┃ turns ┃ tool calls ┃ duration ┃ cost    ┃
┡━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ 96cdc84a │ PASS   │ 6     │ 5          │ 13.6s    │ $0.0983 │
│ bcb670e5 │ PASS   │ 6     │ 5          │ 13.8s    │ $0.0973 │
│ ddc8c5e4 │ PASS   │ 5     │ 4          │ 20.0s    │ $0.0871 │
└──────────┴────────┴───────┴────────────┴──────────┴─────────┘
  pass rate 100%  ·  duration 15.8s ±3.6s  ·  cost per success $0.0942
```

Same prompt, same sandbox image – yet run `ddc8c5e4` took a different path than `96cdc84a`. The aligned trajectory diff shows exactly how they differed (`=` matched, `≈` same action with cosmetic differences, `-`/`+` only in one run):

```
$ flightrec diff runs/fix-median-bug/96cdc84a/trace.jsonl runs/fix-median-bug/ddc8c5e4/trace.jsonl

 ≈ Bash({"command": "ls -la && python3 test_stats.py 2>&1 | head -60"})  [98% similar]
 = Read({"file_path": "/home/user/workspace/stats.py"})
 - Edit({"file_path": "stats.py", "new_string": "def mean(values):\n    if not values:…
 - Edit({"file_path": "stats.py", "new_string": "    return (ordered[mid - 1] + ordered[mid]) / 2…
 + Edit({"file_path": "stats.py", "new_string": "def mean(values):\n    if not values:…
 ≈ Bash({"command": "python3 test_stats.py"})  [94% similar]

3 matched · 2 only in A · 1 only in B
```

One run fixed the two bugs with two separate edits, the other with a single combined edit. On a trivial task this is a curiosity; on a production agent, the same view shows you the exact step where failing runs leave the happy path.

And the same task under two models – reliability vs cost, measured instead of guessed:

```
$ flightrec compare tasks/fix-median-bug.yaml -m sonnet -m haiku -n 3

┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ model  ┃ pass rate  ┃ flaky ┃ duration    ┃ tool calls ┃ cost per success ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ sonnet │ 3/3 (100%) │ no    │ 14.0s ±3.0s │ 5.0        │ $0.0909          │
│ haiku  │ 3/3 (100%) │ no    │ 23.0s ±1.5s │ 6.7        │ $0.0381          │
└────────┴────────────┴───────┴─────────────┴────────────┴──────────────────┘
```

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

# same task under different models: reliability vs cost trade-off
uv run flightrec compare tasks/fix-median-bug.yaml -m sonnet -m haiku -n 3

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
