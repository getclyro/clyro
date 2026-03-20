# Clyro SDK

[![PyPI version](https://img.shields.io/pypi/v/clyro.svg)](https://pypi.org/project/clyro/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/getclyro/clyro/actions/workflows/ci.yml/badge.svg)](https://github.com/getclyro/clyro/actions/workflows/ci.yml)

**Runtime governance for AI agents — prevent failures before they happen.**

One `pip install`, three tools:

| Component | What it does | CLI |
|-----------|-------------|-----|
| **SDK** | Wrap any Python agent with tracing, cost limits, loop detection, and policy enforcement | `clyro-sdk` |
| **MCP Wrapper** | Govern MCP tool calls in Claude Desktop, Cursor, and VS Code | `clyro-mcp` |
| **Claude Code Hooks** | Block destructive commands (rm -rf, DROP TABLE) in Claude Code sessions | `clyro-hook` |

## What is Clyro?

Clyro is a governance platform for AI agents. While most tools let you watch agents fail, Clyro stops failures before they happen — catching infinite loops, runaway costs, and policy violations in real time.

**Works fully offline.** No API key required. Install, wrap, and get governance immediately with local YAML policies. Optionally connect to Clyro Cloud for team dashboards, shared policies, and session replay.

The SDK is the integration layer: add `clyro.wrap()` to any Python agent and you get execution tracing, cost tracking, step limits, loop detection, and policy enforcement — all with zero changes to your agent logic. If the SDK encounters an error, it fails open — your agent keeps running.

## Features

- **Works offline**: Local mode with YAML policies — no cloud dependency
- **5 framework adapters**: LangGraph, CrewAI, Claude Agent SDK, Anthropic SDK, Generic
- **Prevention Stack**: Step limits, cost limits, loop detection, business logic guardrails
- **Policy enforcement**: 8 operators, block/allow/require_approval, per-rule fail-open
- **Cost tracking**: Automatic LLM cost calculation for OpenAI and Anthropic models
- **MCP governance**: JSON-RPC proxy for Claude Desktop, Cursor, VS Code
- **Claude Code hooks**: PreToolUse/PostToolUse governance for Bash, Edit, Write
- **Minimal dependencies**: 6 lightweight packages — no heavy ML frameworks, no vendor lock-in
- **Fail-open design**: SDK failures never break your agent

## Quick Start

### Installation

```bash
pip install clyro
```

### 1. SDK — Wrap any Python agent

```python
import clyro
from clyro import ClyroConfig, ExecutionControls

# No API key needed — runs in local mode automatically
wrapped = clyro.wrap(
    your_agent,
    config=ClyroConfig(
        agent_name="my-agent",
        controls=ExecutionControls(
            max_steps=50,
            max_cost_usd=2.0,
            enable_loop_detection=True,
            enable_policy_enforcement=True,
        ),
    ),
)

# Run normally — governance enforced, session summary printed at end
result = wrapped.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

### 2. MCP Wrapper — Govern MCP tool calls

```bash
# Create config
cat > mcp_governance.yaml << 'EOF'
policies:
  - name: block-dangerous-commands
    rules:
      - tool_name: Bash
        conditions:
          - field: command
            operator: contains
            value: "rm -rf"
        decision: block
        message: "Destructive command blocked"
EOF

# Wrap any MCP server
clyro-mcp wrap --config mcp_governance.yaml -- npx @modelcontextprotocol/server-filesystem /tmp
```

### 3. Claude Code Hooks — Govern Claude Code

```json
// In Claude Desktop settings.json
{
  "hooks": {
    "PreToolUse": [{
      "type": "command",
      "command": "clyro-hook evaluate"
    }]
  }
}
```

### Local YAML Policies

Create `~/.clyro/sdk/policies.yaml`:

```yaml
rules:
  - name: cost-cap
    action_type: llm_call
    conditions:
      - field: cost
        operator: max_value
        value: 5.0
    decision: block
    message: "Session cost exceeded $5.00 limit"

  - name: block-dangerous-tool
    action_type: tool_call
    conditions:
      - field: tool_name
        operator: equals
        value: "delete_database"
    decision: block
    message: "Database deletion not allowed"
```

### Connect to Cloud (optional)

```python
# Add API key to enable cloud features: dashboards, team policies, session replay
config = ClyroConfig(
    api_key="cly_live_...",  # Get from clyrohq.com
    agent_name="my-agent",
    controls=ExecutionControls(max_steps=50, max_cost_usd=2.0),
)
```

## Configuration

### Environment Variables

```bash
export CLYRO_API_KEY="cly_live_..."
export CLYRO_ENDPOINT="https://api.clyrohq.com"
export CLYRO_AGENT_NAME="my-agent"
export CLYRO_MAX_STEPS="50"
export CLYRO_MAX_COST_USD="10.0"
```

```python
from clyro import ClyroConfig

config = ClyroConfig.from_env()
clyro.configure(config)
```

### Programmatic Configuration

```python
from clyro import ClyroConfig, ExecutionControls

config = ClyroConfig(
    # Authentication
    api_key="cly_live_...",
    endpoint="https://api.clyrohq.com",

    # Agent identification
    agent_name="my-production-agent",

    # Execution controls
    controls=ExecutionControls(
        max_steps=50,
        max_cost_usd=5.0,
        loop_detection_threshold=3,
        enable_step_limit=True,
        enable_cost_limit=True,
        enable_loop_detection=True,
    ),

    # Local storage
    local_storage_path="~/.clyro/traces.db",
    local_storage_max_mb=100,

    # Sync settings
    sync_interval_seconds=5.0,
    batch_size=100,
    retry_max_attempts=3,

    # Behavior
    fail_open=True,
    capture_inputs=True,
    capture_outputs=True,
    capture_state=True,
)

clyro.configure(config)
```

## Execution Controls

### Step Limits

Prevent runaway agent executions:

```python
from clyro import ClyroConfig, ExecutionControls, StepLimitExceededError

config = ClyroConfig(
    controls=ExecutionControls(max_steps=10)
)

@clyro.wrap(config=config)
def my_agent():
    # Will raise StepLimitExceededError after 10 steps
    pass

try:
    my_agent()
except StepLimitExceededError as e:
    print(f"Agent exceeded {e.limit} steps")
```

### Cost Limits

Control LLM spending:

```python
from clyro import ClyroConfig, ExecutionControls, CostLimitExceededError

config = ClyroConfig(
    controls=ExecutionControls(max_cost_usd=1.0)
)

@clyro.wrap(config=config)
def my_agent():
    # Will raise CostLimitExceededError if cost exceeds $1.00
    pass

try:
    my_agent()
except CostLimitExceededError as e:
    print(f"Cost ${e.current_cost_usd:.4f} exceeded limit ${e.limit_usd:.2f}")
```

### Loop Detection

Detect infinite loops automatically:

```python
from clyro import ClyroConfig, ExecutionControls, LoopDetectedError

config = ClyroConfig(
    controls=ExecutionControls(
        loop_detection_threshold=3,  # Detect after 3 iterations
        enable_loop_detection=True
    )
)

@clyro.wrap(config=config)
def my_agent():
    # Will raise LoopDetectedError if same state repeats 3 times
    pass

try:
    my_agent()
except LoopDetectedError as e:
    print(f"Loop detected: {e.iterations} iterations")
    print(f"State hash: {e.state_hash}")
```

## Cost Tracking

Automatic cost calculation for LLM calls:

```python
from clyro import calculate_cost

# OpenAI response
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
cost = calculate_cost(response)
print(f"Cost: ${cost:.4f}")

# Anthropic response
response = anthropic.messages.create(
    model="claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": "Hello"}]
)
cost = calculate_cost(response)
print(f"Cost: ${cost:.4f}")
```

## Model Selection

Get cost-optimal model recommendations:

```python
from clyro import ModelSelector

selector = ModelSelector()

# Get recommendation for classification task
recommendation = selector.recommend(
    task_type="classification",
    max_cost_usd=0.001
)

print(f"Recommended model: {recommendation['model']}")
print(f"Expected cost: ${recommendation['expected_cost_usd']:.4f}")
print(f"Parameters: {recommendation['params']}")
```

## Session Access

Access session information during execution:

```python
import clyro

@clyro.wrap
def my_agent(query: str) -> str:
    session = clyro.get_session()
    if session:
        print(f"Step: {session.step_number}")
        print(f"Cost: ${session.cumulative_cost:.4f}")
        print(f"Duration: {session.duration_ms}ms")

    return f"Response: {query}"
```

## Local-Only Mode

Run without backend connection:

```python
config = ClyroConfig(
    api_key=None,  # No API key = local-only mode
    local_storage_path="~/.clyro/traces.db"
)

clyro.configure(config)

@clyro.wrap
def my_agent(query: str) -> str:
    return f"Response: {query}"

# Traces stored locally, not synced to backend
result = my_agent("Hello")
```

## Error Handling

The SDK uses fail-open design - errors are logged but don't break your agent:

```python
import clyro
from clyro import ClyroError, TraceError, TransportError

@clyro.wrap
def my_agent():
    # Even if tracing fails, your agent continues
    return "Success"

try:
    result = my_agent()
except ClyroError as e:
    # SDK errors are caught internally with fail_open=True
    # But you can catch them if needed
    print(f"SDK error: {e}")
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `StepLimitExceededError` raised unexpectedly | `max_steps` set too low for your agent's workload | Increase `max_steps` in `ExecutionControls` or set `enable_step_limit=False` to disable |
| `CostLimitExceededError` on first run | Default cost limit too low for the model you're using | Increase `max_cost_usd` — check `session.cumulative_cost` after a test run to calibrate |
| `LoopDetectedError` false positive | Agent legitimately revisits similar states | Raise `loop_detection_threshold` (default: 3) or disable with `enable_loop_detection=False` |
| Traces not appearing in dashboard | Sync worker hasn't flushed yet, or API key is invalid | Check `CLYRO_API_KEY` is set; traces flush every `sync_interval_seconds` (default: 5s). Inspect `~/.clyro/traces.db` for local buffered traces |
| `TransportError` on startup | Backend unreachable (network issue or wrong endpoint) | Verify `CLYRO_ENDPOINT`; SDK fails open so your agent still runs — traces buffer locally |
| Import error: `ModuleNotFoundError: clyro` | SDK not installed in active environment | Run `pip install clyro` in your virtualenv |
| Agent runs but no traces captured | `@clyro.wrap` decorator missing or `clyro.configure()` not called | Ensure `clyro.configure(config)` runs before any wrapped function is called |
| High memory usage | Large `local_storage_max_mb` or many un-synced traces | Lower `local_storage_max_mb` or check that background sync is running (backend reachable) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Your Agent                             │
│                    (any Python callable)                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ @clyro.wrap
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Clyro SDK Wrapper                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Session    │  │  Transport   │  │    Config    │     │
│  │ Management   │  │    Layer     │  │   Manager    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘     │
│         │                  │                                 │
│         ▼                  ▼                                 │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ TraceEvent   │  │  Background  │                        │
│  │  Creation    │  │  Sync Worker │                        │
│  └──────┬───────┘  └──────┬───────┘                        │
│         │                  │                                 │
│         └──────────┬───────┘                                 │
│                    ▼                                         │
│         ┌──────────────────────┐                            │
│         │  SQLite Local Store  │                            │
│         │  ~/.clyro/traces.db  │                            │
│         └──────────┬───────────┘                            │
│                    │                                         │
└────────────────────┼─────────────────────────────────────────┘
                     │
                     │ HTTPS (background sync)
                     ▼
          ┌──────────────────────┐
          │   Clyro Backend API  │
          │   (PostgreSQL +      │
          │    ClickHouse)       │
          └──────────────────────┘
```

## Framework Adapters

| Framework | Adapter | How it works |
|-----------|---------|-------------|
| **LangGraph** | `LangGraphCallbackHandler` | Node/edge capture, LLM + tool tracing |
| **CrewAI** | `CrewAICallbackHandler` | Task tracing, delegation, inter-agent comms |
| **Claude Agent SDK** | `HookRegistrar` | Hook-based instrumentation, subagent tracking |
| **Anthropic SDK** | Proxy wrapper | Transparent tracing for `messages.create/stream` |
| **Any Python callable** | `@clyro.wrap` | Generic adapter, works with sync/async |

## Documentation

- [API Reference](https://docs.clyrohq.com/sdk) — Full API documentation
- [CHANGELOG](CHANGELOG.md) — Version history
- [CONTRIBUTING](CONTRIBUTING.md) — Development setup and guidelines

## Development

```bash
# Clone and install
git clone https://github.com/getclyro/clyro.git
cd clyro
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=clyro --cov-report=term-missing

# Lint and format
ruff check clyro/
ruff format clyro/
```

### Project Structure

```
clyro/
├── adapters/           # Framework adapters (LangGraph, CrewAI, Anthropic, Claude Agent SDK)
├── mcp/                # MCP governance wrapper (JSON-RPC proxy, YAML policies)
├── hooks/              # Claude Code hooks (PreToolUse/PostToolUse governance)
├── backend/            # Cloud backend communication (HTTP client, sync, circuit breaker)
├── storage/            # Local SQLite storage + migrations
├── workers/            # Background sync workers
├── config.py           # Configuration models (ClyroConfig, ExecutionControls)
├── wrapper.py          # Core wrap() function
├── local_policy.py     # Local YAML policy evaluator
├── local_logger.py     # Terminal logger for local mode
├── cli.py              # CLI (clyro-sdk feedback, help)
├── exceptions.py       # Exception hierarchy
├── cost.py             # LLM cost calculation
└── redaction.py        # PII/secret redaction
tests/
├── sdk/                # SDK unit tests
├── mcp/                # MCP wrapper tests
├── hooks/              # Claude Code hooks tests
└── integration/        # End-to-end tests
```

## Requirements

- Python 3.11+
- httpx, pydantic, structlog, tenacity, aiosqlite, pyyaml

## License

[Apache License 2.0](LICENSE)

## Support Links

- Documentation: https://docs.clyrohq.com
- Issues: https://github.com/getclyro/clyro/issues
- Community: https://discord.gg/clyro
