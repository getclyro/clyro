# Clyro SDK — Claude Agent SDK Usage Guide

Add observability and execution controls to Claude Agent SDK agents with a single line. Clyro hooks into the Claude Agent SDK's hook system to capture tool calls, prompts, and cost estimates — without changing how your agent works.

---

## Install

```bash
pip install clyro claude-agent-sdk
```

Requires Python ≥ 3.11.

---

## Integration

**Before:**
```python
from claude_agent_sdk import Agent, ClaudeAgentOptions

options = ClaudeAgentOptions(
    model="claude-sonnet-4-20250514",
    prompt="You are a helpful assistant.",
)
agent = Agent(options=options)

result = await agent.run("What is the weather in San Francisco?")
```

**After:**
```python
import os
import clyro
from claude_agent_sdk import Agent, ClaudeAgentOptions

clyro.configure(clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-claude-agent",
))

options = ClaudeAgentOptions(
    model="claude-sonnet-4-20250514",
    prompt="You are a helpful assistant.",
)
agent = Agent(options=options)

wrapped = clyro.wrap(agent)  # one line added

result = await wrapped("What is the weather in San Francisco?")  # same call
```

Clyro automatically registers hooks into the Claude Agent SDK. It captures tool calls (PreToolUse/PostToolUse), user prompts, subagent starts, and session lifecycle — no other changes needed.

> **Note:** `agent_name` (or `agent_id`) is **required**. The SDK raises `ClyroWrapError` if neither is provided. `agent_name` is used for auto-registration — Clyro generates a stable `agent_id` from it and links all traces to that agent.

### Adapter detection

`clyro.wrap()` auto-detects the Claude Agent SDK adapter when the agent object is a `claude_agent_sdk.Agent` instance (checks for `claude_agent_sdk` in the module name, or `hooks` + `model` attributes). If auto-detection fails (e.g., the agent is wrapped in another object), pass the adapter explicitly:

```python
wrapped = clyro.wrap(agent, adapter="claude_agent_sdk")
```

Without the correct adapter, Clyro falls back to `generic` — which only traces the top-level call and misses tool calls, prompts, subagent starts, and other Claude Agent SDK-specific events.

---

## What Gets Traced

| Event | When |
|---|---|
| `session_start` | Agent session begins |
| `tool_call` | PreToolUse — tool is about to be called |
| `tool_result` | PostToolUse — tool execution completed |
| `user_prompt_submit` | User prompt submitted to the agent |
| `subagent_start` | A subagent is spawned |
| `error` | Any exception raised (including policy violations) |
| `session_end` | Agent session completes |

---

## Configuration

### Option A — Environment variables

```bash
export CLYRO_API_KEY="your-clyro-api-key"
export CLYRO_AGENT_NAME="my-claude-agent"
export CLYRO_MAX_STEPS="50"
export CLYRO_MAX_COST_USD="10.0"
```

```python
import clyro

clyro.configure(clyro.ClyroConfig.from_env())

wrapped = clyro.wrap(agent)
result = await wrapped("What is the weather?")
```

### Option B — Programmatic config

```python
import os
import clyro

clyro.configure(clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-claude-agent",
))

wrapped = clyro.wrap(agent)
```

### Option C — Per-agent config

```python
import os
import clyro

config = clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-claude-agent",
)

wrapped = clyro.wrap(agent, config=config)
```

---

## Execution Controls

```python
import os
from clyro import ClyroConfig, ExecutionControls

config = ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-claude-agent",
    controls=ExecutionControls(
        max_steps=50,                 # raise StepLimitExceededError after 50 tool calls
        max_cost_usd=5.0,             # raise CostLimitExceededError above $5
        loop_detection_threshold=3,   # raise LoopDetectedError if same state repeats 3x
    ),
)

wrapped = clyro.wrap(agent, config=config)
```

Catching limit errors:

```python
from clyro import StepLimitExceededError, CostLimitExceededError, LoopDetectedError

try:
    result = await wrapped("do something complex")
except StepLimitExceededError as e:
    print(f"Stopped: exceeded {e.limit} steps")
except CostLimitExceededError as e:
    print(f"Stopped: cost ${e.current_cost_usd:.4f} exceeded limit ${e.limit_usd:.2f}")
except LoopDetectedError as e:
    print(f"Stopped: loop detected after {e.iterations} iterations")
```

---

## Cost Estimation

The Claude Agent SDK spawns the Claude Code CLI as a subprocess. Per-call token counts are not available through the hook interface. Clyro uses a character-length heuristic to estimate costs:

```
estimated_tokens = len(content) / 4
estimated_cost   = estimated_tokens * cost_per_token_usd
```

Content used for estimation per hook:

| Hook | Content |
|---|---|
| `PreToolUse` | Tool call arguments (`json.dumps(tool_input)`) |
| `PostToolUse` | Tool execution result (`json.dumps(tool_response)`) |
| `PostToolUseFailure` | Error message |
| `UserPromptSubmit` | User prompt text |

All costs are marked with `cost_estimated: True` in trace event metadata.

See [costing_limitations.md](../../limitations/adapter-limitations/costing_limitations.md) for details.

---

## Policy Enforcement

The Claude Agent SDK adapter evaluates policies **pre-action** — before tool calls execute. This enables:

- **Cost-based policies**: Block or require approval when estimated cost exceeds thresholds
- **Keyword filters**: Block prompts containing sensitive keywords (`field: input, operator: contains`)
- **Tool parameter rules**: Block tool calls based on parameter values (`field: rmq_cluster, operator: equals`)
- **Approval prompts**: Interactive `[y/n]` prompt for `require_approval` decisions

Policy parameters available for rules:

| Field | Description |
|---|---|
| `tool_name` | Name of the tool being called |
| `input` | User's prompt text |
| `cost` | Estimated cumulative cost |
| `step_number` | Current step count |
| Tool input keys | Flattened tool arguments (e.g., `rmq_cluster`, `site_name`) |

---

## Full Configuration Reference

```python
import os
from clyro import ClyroConfig, ExecutionControls

config = ClyroConfig(
    # Authentication
    api_key=os.environ.get("CLYRO_API_KEY"),           # or use CLYRO_API_KEY env var + from_env()
    endpoint="https://api.clyrohq.com",  # or set CLYRO_ENDPOINT env var

    # Agent identification
    agent_name="my-claude-agent",     # human-readable name; auto-registers on first run

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
| `CLYRO_ENDPOINT` | `endpoint` | `https://api.clyrohq.com` |
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

wrapped = clyro.wrap(agent)
```

---

## Failure Behavior

The SDK is fail-open (`fail_open=True` by default). If Clyro's backend is unreachable or tracing encounters an error, your agent continues executing normally. Events are buffered locally and retried on the next sync interval.

To make tracing failures fatal (useful in tests):

```python
config = ClyroConfig(fail_open=False, ...)
```
