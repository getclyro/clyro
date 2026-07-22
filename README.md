# Clyro SDK

[![PyPI version](https://img.shields.io/pypi/v/clyro.svg)](https://pypi.org/project/clyro/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/getclyro/clyro/actions/workflows/ci.yml/badge.svg)](https://github.com/getclyro/clyro/actions/workflows/ci.yml)

**Runtime governance for AI agents: prevent failures before they happen.**

One `pip install`, three tools:

| Component | What it does | CLI |
|-----------|-------------|-----|
| **SDK** | Wrap any Python agent with tracing, cost limits, loop detection, and policy enforcement | `clyro` / `clyro-sdk` |
| **Policy Recommender** | Point it at an existing agent → recommends the agent type, concerns, and kits to govern | `clyro suggest` |
| **MCP Wrapper** | Govern MCP tool calls in Claude Desktop, Cursor, and VS Code | `clyro-mcp` |
| **Claude Code Hooks** | Block destructive commands (rm -rf, DROP TABLE) in Claude Code sessions | `clyro-hook` |

## What is Clyro?

Clyro is a governance platform for AI agents. While most tools let you watch agents fail, Clyro stops failures before they happen, catching infinite loops, runaway costs, and policy violations in real time.

**Works fully offline.** No API key required. Install, wrap, and get governance immediately with local YAML policies. Optionally connect to Clyro Cloud for team dashboards, shared policies, and session replay.

The SDK is the integration layer: add `clyro.wrap()` to any Python agent and you get execution tracing, cost tracking, step limits, loop detection, and policy enforcement, all with zero changes to your agent logic. If the SDK encounters an error it fails open, and your agent keeps running.

## Features

- **Works offline**: Local mode with YAML policies, no cloud dependency
- **5 framework adapters**: LangGraph, CrewAI, Claude Agent SDK, Anthropic SDK, Generic
- **Policy recommender** (`clyro suggest`): point it at an existing agent → it recommends the agent type, concerns, and kits to govern
- **Prevention Stack**: Step limits, cost limits, loop detection, business logic guardrails
- **Policy enforcement**: 8 operators, block/allow/require_approval, per-rule fail-open
- **Cost tracking**: Automatic LLM cost calculation for OpenAI and Anthropic models
- **MCP governance**: JSON-RPC proxy for Claude Desktop, Cursor, VS Code
- **Claude Code hooks**: PreToolUse/PostToolUse governance for Bash, Edit, Write
- **Minimal dependencies**: 6 lightweight packages, no heavy ML frameworks, no vendor lock-in
- **Fail-open design**: SDK failures never break your agent

## Quick Start

### Installation

```bash
pip install clyro
```

### 1. SDK: Wrap any Python agent

```python
import clyro
from clyro import ClyroConfig, ExecutionControls

# No API key needed: runs in local mode automatically
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

# Run normally: governance enforced, session summary printed at end
result = wrapped.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

### 2. MCP Wrapper: Govern MCP tool calls

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

### 3. Claude Code Hooks: Govern Claude Code

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

### 4. Policy Recommender: what should I govern?

Point Clyro at an agent you've already built and it recommends the agent type,
the concerns worth tracking, and the kits to apply, reading its tools, prompt,
and structure (it never runs the agent). Runs locally, no account needed.

```bash
clyro suggest myapp.agents:support_agent
```
```
Detected agent type: agent_type.transactional
Inferred concerns:
  • concern.pii-protection   [high]   — Tool argument `email` is PII.
  • concern.reversibility    [high]   — Tool `refund_customer` performs an irreversible action.
```

`--json` for the machine-readable payload, `--llm-transport rule-based` for a
deterministic offline run. Full guide: [docs/sdk/policy-recommender.md](docs/sdk/policy-recommender.md).

### Local YAML Policies

Create `~/.clyro/sdk/policies.yaml`:

```yaml
version: 1
default_action: allow            # required; decision when no rule matches

actions:
  llm_call:
    policies:
      - name: cost-cap
        parameter: cost
        operator: max_value      # matches when cost > 5.0
        value: 5.0
        action: block            # matched → block (action is required)

  tool_call:
    policies:
      - name: block-dangerous-tool
        parameter: tool_name
        operator: equals         # matches when tool_name == "delete_database"
        value: "delete_database"
        action: block
```

Both `default_action` (root) and per-rule `action` are **required**. Each rule
fires its `action` when its condition matches; `default_action` is the
fallback when no rule matches.

### Connect to Cloud (optional)

```python
# Add API key to enable cloud features: dashboards, team policies, session replay
config = ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),  # Get from clyro.dev
    agent_name="my-agent",
    controls=ExecutionControls(max_steps=50, max_cost_usd=2.0),
)
```

> **Cloud mode: the dashboard's `default_action` always wins.** When you set
> an `api_key` (cloud mode), the cloud dashboard's `default_action` is
> authoritative: your local YAML's `default_action` is treated as a
> fallback that applies only when the agent has no cloud policies attached
> (or the policy fetch fails). This is **cloud-wins** precedence: a
> centrally-mandated default cannot be silently overridden by an
> out-of-date local config.
>
> This applies to the `default_action` fallback only. Explicit local rules
> (rules whose conditions match) still fast-fail pre-flight as before.
> To honor the local YAML's `default_action`, run without `api_key`
> (local-only mode).

## Configuration

### Environment Variables

```bash
export CLYRO_API_KEY="your-clyro-api-key"
export CLYRO_ENDPOINT="https://api.clyro.dev"
export CLYRO_AGENT_NAME="my-agent"
export CLYRO_MAX_STEPS="50"
export CLYRO_MAX_COST_USD="10.0"

# Monitor mode: evaluate every check, block nothing (see "Dry-Run Mode" below).
# Truthy: true, 1, yes, on, dry_run. Any other value set = enforce (fail-safe).
export CLYRO_DRY_RUN="true"
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
    api_key=os.environ.get("CLYRO_API_KEY"),
    endpoint="https://api.clyro.dev",

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
        # "enforce" (default) or "dry_run"; see "Dry-Run Mode" below.
        enforcement_mode="enforce",
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

## Dry-Run Mode

Run the full governance stack without enforcing it. Every check still evaluates and
records what it *would* have blocked, but nothing is ever stopped. Use it to validate
limits and policies against real traffic before turning enforcement on.

```python
controls=ExecutionControls(
    max_steps=25,
    max_cost_usd=10.0,
    enable_policy_enforcement=True,
    enforcement_mode="dry_run",   # monitor only; nothing is blocked
)
```

Or without changing code:

```bash
CLYRO_DRY_RUN=true python my_agent.py
```

You'll see a banner once at startup, then one line per distinct finding:

```
CLYRO-DRYRUN active — enforcement suppressed (mode=dry_run)   surface=sdk
CLYRO-DRYRUN would-have-blocked  action=step_26  check=step  rule=None  would_be=block
```

Each finding is also recorded as a trace event with `event_type="would_block"`,
de-duplicated to one marker per distinct reason: a rule that trips on 500 actions
records one event, not 500.

**What dry-run does not suppress:**

- **Absolute ceilings** (`absolute_max_steps`, default 1,000,000; `absolute_max_cost_usd`,
  default $100,000) still fire. They are the runaway-agent backstop and cannot be disabled
  or relaxed by dry-run. This holds on **all three surfaces**: the SDK raises
  `AbsoluteCeilingExceededError`; the MCP wrapper returns a hard JSON-RPC error and does not
  forward the call; the Claude Code hook returns a real block so the tool does not run.
  (SDK: `ExecutionControls.absolute_max_*`; MCP/hooks: `global.absolute_max_*` in the config file.)
- **Approval handlers are skipped**: a `require_approval` policy records a marker and
  proceeds, so an unattended run cannot hang waiting on a prompt.

Dry-run events are marked at write time and excluded from all analytics read paths, so
a monitor-mode session never moves your Reliability Score or dashboard metrics.

The same flag exists on the other surfaces: `dry_run: true` (top level) in the MCP
wrapper and Claude Code hooks configs, plus `clyro-mcp wrap --dry-run`. Precedence is
`--dry-run` > `CLYRO_DRY_RUN` > config file. The CLI flag is MCP-only, so for the SDK
and hooks the chain is `CLYRO_DRY_RUN` > config.

Full guide: [Dry-Run Mode](https://docs.clyro.dev/docs/concepts/dry-run-mode).

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
| `CostLimitExceededError` on first run | Default cost limit too low for the model you're using | Increase `max_cost_usd`; check `session.cumulative_cost` after a test run to calibrate |
| `LoopDetectedError` false positive | Agent legitimately revisits similar states | Raise `loop_detection_threshold` (default: 3) or disable with `enable_loop_detection=False` |
| Traces not appearing in dashboard | Sync worker hasn't flushed yet, or API key is invalid | Check `CLYRO_API_KEY` is set; traces flush every `sync_interval_seconds` (default: 5s). Inspect `~/.clyro/traces.db` for local buffered traces |
| `TransportError` on startup | Backend unreachable (network issue or wrong endpoint) | Verify `CLYRO_ENDPOINT`; SDK fails open so your agent still runs; traces buffer locally |
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

### Usage Guides

| Guide | Description |
|-------|-------------|
| [Policy Recommender](docs/sdk/policy-recommender.md) | `clyro suggest`; recommend the agent type, concerns, and kits to govern |
| [LangGraph](docs/sdk/langgraph.md) | Wrap LangGraph agents with governance |
| [CrewAI](docs/sdk/crewai.md) | Wrap CrewAI agents with governance |
| [Claude Agent SDK](docs/sdk/claude_agent_sdk.md) | Wrap Claude Agent SDK with governance |
| [Anthropic SDK](docs/sdk/anthropic.md) | Wrap Anthropic SDK calls with governance |
| [MCP Wrapper](docs/mcp/mcp_wrapper.md) | Govern MCP tool calls in Claude Desktop, Cursor, VS Code |
| [Claude Code Hooks](docs/hooks/claude_code_hooks.md) | Block destructive commands in Claude Code |
| [OpenTelemetry](docs/otel/opentelemetry.md) | Export traces to OTLP-compatible backends |
| [CX Policy](docs/policy/cx_policy.md) | Configure customer experience policies |

### Reference

- [API Reference](https://docs.clyro.dev/sdk): Full API documentation
- [CHANGELOG](CHANGELOG.md): Version history
- [CONTRIBUTING](CONTRIBUTING.md): Development setup and guidelines

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
├── recommender/        # Policy recommender (`clyro suggest`): introspection, mappers, transports
├── mcp/                # MCP governance wrapper (JSON-RPC proxy, YAML policies)
├── hooks/              # Claude Code hooks (PreToolUse/PostToolUse governance)
├── backend/            # Cloud backend communication (HTTP client, sync, circuit breaker)
├── storage/            # Local SQLite storage + migrations
├── workers/            # Background sync workers
├── config.py           # Configuration models (ClyroConfig, ExecutionControls)
├── wrapper.py          # Core wrap() function
├── local_policy.py     # Local YAML policy evaluator
├── local_logger.py     # Terminal logger for local mode
├── cli.py              # CLI (clyro / clyro-sdk: suggest, feedback, status, help)
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

- Documentation: https://docs.clyro.dev
- Issues: https://github.com/getclyro/clyro/issues
- Community: https://discord.gg/clyro
