# Clyro SDK — Anthropic Python SDK Usage Guide

Add observability and execution controls to Anthropic API calls with a single line. Clyro wraps your `Anthropic` client to capture LLM calls with token counts, tool usage, cost tracking, and policy enforcement — without changing how your code works.

---

## Install

```bash
pip install clyro anthropic
```

Requires Python ≥ 3.11. Requires Anthropic SDK ≥ 0.18.0.

---

## Integration

**Before:**
```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
)
print(response.content[0].text)
```

**After:**
```python
import os
import clyro
from anthropic import Anthropic

clyro.configure(clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-anthropic-agent",
))

client = Anthropic()

traced = clyro.wrap(client)  # one line added

response = traced.messages.create(  # same call
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
)
print(response.content[0].text)

traced.close()  # flush remaining events
```

Clyro returns a transparent proxy that intercepts `messages.create()` and `messages.stream()` calls. It captures token counts, calculates costs, detects tool use, and enforces execution controls — all other client methods pass through unchanged.

> **Note:** `agent_name` (or `agent_id`) is **required**. The SDK raises `ClyroWrapError` if neither is provided. `agent_name` is used for auto-registration — Clyro generates a stable `agent_id` from it and links all traces to that agent.

### Adapter detection

`clyro.wrap()` auto-detects the Anthropic adapter when the client object is an `anthropic.Anthropic` or `anthropic.AsyncAnthropic` instance (checks for `anthropic` in the module name). If auto-detection fails (e.g., the client is wrapped in another object), pass the adapter explicitly:

```python
traced = clyro.wrap(client, adapter="anthropic")
```

Without the correct adapter, Clyro falls back to `generic` — which only traces the top-level call and misses LLM call details, tool use detection, cost tracking, and other Anthropic-specific events.

---

## What Gets Traced

| Event | When |
|---|---|
| `session_start` | First API call (lazy initialization) |
| `llm_call` | Each `messages.create()` or `messages.stream()` completes (with model, tokens, cost, duration) |
| `tool_call` | For each `tool_use` block in the response (with tool name, input, and tool_use_id) |
| `step` | Per `create()`/`stream()` call (with step number and agent stage) |
| `error` | API errors, enforcement failures, or policy violations |
| `session_end` | On `close()` or auto-flush (with cumulative cost) |

---

## Async and Streaming

### Async client

```python
import clyro
from anthropic import AsyncAnthropic

client = AsyncAnthropic()
traced = clyro.wrap(client)

response = await traced.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Streaming

```python
with traced.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a poem"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

Clyro wraps the stream context manager and emits `llm_call` and `tool_call` events on stream completion.

---

## Agentic Tool-Use Loop

When building tool-use loops with the Anthropic SDK, Clyro traces every iteration:

```python
import os
import clyro
from anthropic import Anthropic

traced = clyro.wrap(Anthropic(), config=clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="tool-use-agent",
))

messages = [{"role": "user", "content": "What's the weather in SF and NYC?"}]
tools = [{"name": "get_weather", "description": "Get weather", "input_schema": {...}}]

while True:
    response = traced.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=messages,
        tools=tools,
    )

    # Clyro automatically traces each LLM call, detects tool_use blocks,
    # emits TOOL_CALL events, and enforces step/cost/policy limits

    if response.stop_reason == "end_turn":
        break

    # Process tool calls and append results
    messages.append({"role": "assistant", "content": response.content})
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = call_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
    messages.append({"role": "user", "content": tool_results})

traced.close()
```

---

## Configuration

### Option A — Environment variables

```bash
export CLYRO_API_KEY="your-clyro-api-key"
export CLYRO_AGENT_NAME="my-anthropic-agent"
export CLYRO_MAX_STEPS="50"
export CLYRO_MAX_COST_USD="10.0"
```

```python
import clyro

clyro.configure(clyro.ClyroConfig.from_env())

traced = clyro.wrap(client)
response = traced.messages.create(...)
```

### Option B — Programmatic config

```python
import os
import clyro

clyro.configure(clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-anthropic-agent",
))

traced = clyro.wrap(client)
```

### Option C — Per-agent config

```python
import os
import clyro

config = clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-anthropic-agent",
)

traced = clyro.wrap(client, config=config)
```

---

## Execution Controls

```python
import os
from clyro import ClyroConfig, ExecutionControls

config = ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-anthropic-agent",
    controls=ExecutionControls(
        max_steps=50,                 # raise StepLimitExceededError after 50 API calls
        max_cost_usd=5.0,             # raise CostLimitExceededError above $5
        loop_detection_threshold=3,   # raise LoopDetectedError if same message repeats 3x
    ),
)

traced = clyro.wrap(client, config=config)
```

Catching limit errors:

```python
from clyro import StepLimitExceededError, CostLimitExceededError, LoopDetectedError

try:
    response = traced.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=messages,
        tools=tools,
    )
except StepLimitExceededError as e:
    print(f"Stopped: exceeded {e.limit} steps")
except CostLimitExceededError as e:
    print(f"Stopped: cost ${e.current_cost_usd:.4f} exceeded limit ${e.limit_usd:.2f}")
except LoopDetectedError as e:
    print(f"Stopped: loop detected after {e.iterations} iterations")
```

---

## Policy Enforcement

The Anthropic adapter evaluates policies **pre-action** — before tool calls execute. This enables:

- **Cost-based policies**: Block or require approval when cumulative cost exceeds thresholds
- **Tool parameter rules**: Block tool calls based on parameter values (`field: rmq_cluster, operator: equals`)
- **Approval prompts**: Interactive `[y/n]` prompt for `require_approval` decisions

Policy parameters available for rules:

| Field | Description |
|---|---|
| `tool_name` | Name of the tool being called |
| `cost` | Cumulative cost so far |
| `step_number` | Current step count |
| Tool input keys | Flattened tool arguments (e.g., `rmq_cluster`, `site_name`) |

---

## Cost Tracking

The Anthropic adapter extracts **real token counts** from API responses (not estimates). Costs are calculated from the model's per-token pricing:

| Field | Source |
|---|---|
| `input_tokens` | `response.usage.input_tokens` |
| `output_tokens` | `response.usage.output_tokens` |
| `cost_usd` | Calculated from token counts using model-specific rates |

Cost is accumulated across all `messages.create()` / `messages.stream()` calls within a session and used for cost limit enforcement.

---

## Event Hierarchy

Clyro maintains parent-child relationships between events within a session:

- Each `llm_call` event references the previous `llm_call` as its parent
- Each `tool_call` event references the `llm_call` that generated it as its parent

This enables tracing the full chain of reasoning and tool use in multi-turn conversations.

---

## Full Configuration Reference

```python
import os
from clyro import ClyroConfig, ExecutionControls

config = ClyroConfig(
    # Authentication
    api_key=os.environ.get("CLYRO_API_KEY"),           # or use CLYRO_API_KEY env var + from_env()
    endpoint="https://api.clyro.dev",  # or set CLYRO_ENDPOINT env var

    # Agent identification
    agent_name="my-anthropic-agent",  # human-readable name; auto-registers on first run

    # Execution controls
    controls=ExecutionControls(
        max_steps=100,
        max_cost_usd=10.0,
        loop_detection_threshold=3,
        enable_step_limit=True,
        enable_cost_limit=True,
        enable_loop_detection=True,
        enable_policy_enforcement=True,
    ),

    # Local storage (events are buffered here before sync)
    local_storage_path="~/.clyro/traces.db",
    local_storage_max_mb=100,

    # Backend sync
    sync_interval_seconds=5.0,        # how often to flush buffered events
    batch_size=100,
    retry_max_attempts=3,

    # Behaviour
    fail_open=True,                   # if tracing fails, agent continues normally
    capture_inputs=True,
    capture_outputs=True,
)
```

### Environment variables (used with `ClyroConfig.from_env()`)

| Variable | Config equivalent | Default |
|---|---|---|
| `CLYRO_API_KEY` | `api_key` | — |
| `CLYRO_ENDPOINT` | `endpoint` | `https://api.clyro.dev` |
| `CLYRO_AGENT_NAME` | `agent_name` | — |
| `CLYRO_MAX_STEPS` | `controls.max_steps` | `100` |
| `CLYRO_MAX_COST_USD` | `controls.max_cost_usd` | `10.0` |

---

## Local-only Mode

```python
import clyro

clyro.configure(clyro.ClyroConfig(
    api_key=None,                               # no key = no sync
    local_storage_path="~/.clyro/traces.db",
))

traced = clyro.wrap(client)
```

---

## Failure Behavior

The SDK is fail-open (`fail_open=True` by default). If Clyro's backend is unreachable or tracing encounters an error, your Anthropic client continues executing normally. Events are buffered locally and retried on the next sync interval.

To make tracing failures fatal (useful in tests):

```python
config = ClyroConfig(fail_open=False, ...)
```
