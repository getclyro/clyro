# Clyro SDK — LangGraph Usage Guide

Add observability and execution controls to LangGraph agents with a single line. Clyro captures node executions, LLM calls, tool calls, retriever calls (RAG), and state transitions — without changing how your graph works.

---

## Install

```bash
pip install clyro
```

Requires Python ≥ 3.11.

---

## Integration

**Before:**
```python
from langgraph.graph import StateGraph

graph = StateGraph(MyState)
graph.add_node("researcher", research_node)
graph.add_node("writer", write_node)
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "__end__")
compiled = graph.compile()

result = compiled.invoke({"topic": "climate change"})
```

**After:**
```python
import os
import clyro
from langgraph.graph import StateGraph

clyro.configure(clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-graph",
))

graph = StateGraph(MyState)
graph.add_node("researcher", research_node)
graph.add_node("writer", write_node)
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "__end__")
compiled = graph.compile()

wrapped = clyro.wrap(compiled)  # one line added

result = wrapped.invoke({"topic": "climate change"})  # same call
```

Clyro automatically injects LangChain callbacks. It captures node executions, LLM calls, tool calls, and retriever calls (RAG) — no other changes needed.

> **Note:** `agent_name` (or `agent_id`) is **required**. The SDK raises `ClyroWrapError` if neither is provided. `agent_name` is used for auto-registration — Clyro generates a stable `agent_id` from it and links all traces to that agent.

---

## What Gets Traced

| Event | When |
|---|---|
| `session_start` | Wrapper begins executing |
| `llm_call` | LLM is invoked (with token counts and cost) |
| `tool_call` | A tool or function is called |
| `retriever_call` | A RAG retriever is queried |
| `state_transition` | LangGraph moves between nodes |
| `error` | Any exception raised |
| `session_end` | Execution completes (with total steps and cost) |

---

## Configuration

### Option A — Environment variables

```bash
export CLYRO_API_KEY="your-clyro-api-key"
export CLYRO_AGENT_NAME="my-graph"
export CLYRO_MAX_STEPS="50"
export CLYRO_MAX_COST_USD="10.0"
```

```python
import clyro

clyro.configure(clyro.ClyroConfig.from_env())

wrapped = clyro.wrap(compiled)
result = wrapped.invoke({"topic": "climate change"})
```

### Option B — Programmatic config

```python
import os
import clyro

clyro.configure(clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-graph",
))

wrapped = clyro.wrap(compiled)
```

### Option C — Per-agent config

```python
import os
import clyro

config = clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-graph",
)

wrapped = clyro.wrap(compiled, config=config)
```

---

## Execution Controls

```python
import os
from clyro import ClyroConfig, ExecutionControls

config = ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-graph",
    controls=ExecutionControls(
        max_steps=50,                 # raise StepLimitExceededError after 50 tool calls
        max_cost_usd=5.0,             # raise CostLimitExceededError above $5
        loop_detection_threshold=3,   # raise LoopDetectedError if same state repeats 3x
    ),
)

wrapped = clyro.wrap(compiled, config=config)
```

Catching limit errors:

```python
from clyro import StepLimitExceededError, CostLimitExceededError, LoopDetectedError

try:
    result = wrapped.invoke({"topic": "climate change"})
except StepLimitExceededError as e:
    print(f"Stopped: exceeded {e.limit} steps")
except CostLimitExceededError as e:
    print(f"Stopped: cost ${e.current_cost_usd:.4f} exceeded limit ${e.limit_usd:.2f}")
except LoopDetectedError as e:
    print(f"Stopped: loop detected after {e.iterations} iterations")
```

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
    agent_name="my-graph",            # human-readable name; auto-registers on first run

    # Execution controls
    controls=ExecutionControls(
        max_steps=100,
        max_cost_usd=10.0,
        loop_detection_threshold=3,
        enable_step_limit=True,
        enable_cost_limit=True,
        enable_loop_detection=True,
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
    capture_state=True,
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

Run without sending any data to Clyro. Traces are stored locally in SQLite.

```python
import clyro

clyro.configure(clyro.ClyroConfig(
    api_key=None,                               # no key = no sync
    local_storage_path="~/.clyro/traces.db",
))

wrapped = clyro.wrap(compiled)
```

---

## Accessing Session State

```python
import clyro

wrapped = clyro.wrap(compiled)

# Inside your node function:
def research_node(state):
    session = clyro.get_session()
    if session:
        print(f"step {session.step_number} | cost ${session.cumulative_cost:.4f}")
    # ...
```

---

## Failure Behavior

The SDK is fail-open (`fail_open=True` by default). If Clyro's backend is unreachable or tracing encounters an error, your agent continues executing normally. Events are buffered locally and retried on the next sync interval.

To make tracing failures fatal (useful in tests):

```python
config = ClyroConfig(fail_open=False, ...)
```
