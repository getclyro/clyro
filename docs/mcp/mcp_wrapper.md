# Clyro MCP Wrapper — Usage Guide

A transparent governance proxy for MCP servers. It sits between your AI host (Claude Desktop, Cursor, Continue, any MCP client) and your MCP server — enforcing policies, tracking costs, detecting loops, and syncing traces to Clyro — without any changes to either side.

```
[AI Host]  ←─ stdio ─→  [clyro-mcp wrap]  ←─ stdio ─→  [Real MCP Server]
```

Neither the host nor the server knows the wrapper exists.

---

## Install

```bash
pip install clyro
```

This installs the unified Clyro SDK which includes the `clyro-mcp` CLI command.

Requires Python ≥ 3.11.

### Finding the full path to `clyro-mcp`

GUI applications (VS Code, Claude Desktop, Cursor) do **not** load your shell profile (`~/.bashrc`, `~/.zshrc`), so they cannot resolve `clyro-mcp` by name. You must use the **full path** to the executable in their configs.

Find the full path after installation:

```bash
which clyro-mcp
```

Common locations depending on how you installed:

| Install method | Typical path |
|---|---|
| System pip | `/usr/local/bin/clyro-mcp` |
| User pip (`--user`) | `~/.local/bin/clyro-mcp` |
| virtualenv / venv | `/path/to/venv/bin/clyro-mcp` |
| conda env | `~/miniconda3/envs/<env>/bin/clyro-mcp` |
| Editable install | Same as the venv/conda env it was installed into |

> **Tip:** If you want to use the bare `clyro-mcp` command in GUI apps, create a symlink in a directory that is on the system-wide PATH (e.g., `/usr/local/bin`):
> ```bash
> sudo ln -s $(which clyro-mcp) /usr/local/bin/clyro-mcp
> ```

---

## One-liner Integration

The only change needed is in how your host launches the MCP server. Prepend `clyro-mcp wrap` to the existing server command.

**Before:**
```
npx @modelcontextprotocol/server-filesystem /home/user/projects
```

**After:**
```
clyro-mcp wrap npx @modelcontextprotocol/server-filesystem /home/user/projects
```

That's it. No changes to the MCP server. No changes to your application code.

---

## Integration by Host

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

**Before:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    }
  }
}
```

**After:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "/path/to/venv/bin/clyro-mcp",
      "args": ["wrap", "npx", "@modelcontextprotocol/server-filesystem", "/home/user/projects"],
      "env": {
        "CLYRO_API_KEY": "your-clyro-api-key"
      }
    }
  }
}
```

> **Important:** Replace `/path/to/venv/bin/clyro-mcp` with the actual path from `which clyro-mcp`. GUI apps like Claude Desktop do not load shell profiles, so the bare `clyro-mcp` command will not be found.

### Cursor

Edit `.cursor/mcp.json` in your project or `~/.cursor/mcp.json` globally:

**Before:**
```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["-m", "my_mcp_server"]
    }
  }
}
```

**After:**
```json
{
  "mcpServers": {
    "my-server": {
      "command": "/path/to/venv/bin/clyro-mcp",
      "args": ["wrap", "python", "-m", "my_mcp_server"],
      "env": {
        "CLYRO_API_KEY": "your-clyro-api-key"
      }
    }
  }
}
```

> **Important:** Replace `/path/to/venv/bin/clyro-mcp` with the actual path from `which clyro-mcp`. Cursor does not load shell profiles.

### VS Code (Claude Code)

Edit `~/.claude.json` to add or wrap MCP servers globally:

**Before:**
```json
{
  "mcpServers": {
    "my-server": {
      "type": "stdio",
      "command": "/path/to/venv/bin/mcp-server-fetch",
      "args": [],
      "env": {}
    }
  }
}
```

**After:**
```json
{
  "mcpServers": {
    "my-server": {
      "type": "stdio",
      "command": "/path/to/venv/bin/clyro-mcp",
      "args": [
        "wrap",
        "--config", "/path/to/my_governance.yaml",
        "/path/to/venv/bin/mcp-server-fetch"
      ],
      "env": {}
    }
  }
}
```

> **Important:** Replace `/path/to/venv/bin/clyro-mcp` with the actual path from `which clyro-mcp`. VS Code extensions do not load shell profiles (`~/.bashrc`, `~/.zshrc`), so the bare `clyro-mcp` command will not be found.

The `--config` flag points to a YAML file where you define governance policies, backend settings (`agent_name`, `api_key`, `api_url`), and audit options. See the [Configuration with `--config`](#configuration-with---config) section below.

### Custom Python MCP servers

If you built your own MCP server in Python, the wrapper works the same way — just change the launch command in your host config.

> **Note:** The examples below use the bare `clyro-mcp` command, which works in terminal sessions. If you're using these snippets in a GUI app config (VS Code, Claude Desktop, Cursor), replace `clyro-mcp` with the full path (see [Finding the full path](#finding-the-full-path-to-clyro-mcp)).

**Using `python -m`:**

Before:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["-m", "my_package.server"]
    }
  }
}
```

After:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "clyro-mcp",
      "args": ["wrap", "python", "-m", "my_package.server"],
      "env": {
        "CLYRO_API_KEY": "your-clyro-api-key"
      }
    }
  }
}
```

**Using `uv run` (recommended for project-local dependencies):**

Before:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "uv",
      "args": ["run", "python", "-m", "my_package.server"]
    }
  }
}
```

After:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "clyro-mcp",
      "args": ["wrap", "uv", "run", "python", "-m", "my_package.server"],
      "env": {
        "CLYRO_API_KEY": "your-clyro-api-key"
      }
    }
  }
}
```

**Using a script entrypoint:**

Before:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "uv",
      "args": ["run", "my-server-script"]
    }
  }
}
```

After:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "clyro-mcp",
      "args": ["wrap", "uv", "run", "my-server-script"],
      "env": {
        "CLYRO_API_KEY": "your-clyro-api-key"
      }
    }
  }
}
```

The wrapper passes the working directory and all environment variables through to the child process unchanged, so your server sees the same environment it would see when launched directly.

> **Note:** GUI applications (VS Code, Claude Desktop, Cursor) do not load shell profiles, so always use the full path to `clyro-mcp` in their configs. The bare `clyro-mcp` command works in terminal sessions where your shell profile is loaded. See [Finding the full path to `clyro-mcp`](#finding-the-full-path-to-clyro-mcp) above.

### Any other MCP host

The pattern is always the same. Find where the host launches the MCP server process and prepend `clyro-mcp wrap`:

```
# Original server command
my-mcp-server --port 3000 --data /path/to/data

# Wrapped
clyro-mcp wrap my-mcp-server --port 3000 --data /path/to/data
```

### Terminal / manual launch

```bash
export CLYRO_API_KEY="your-clyro-api-key"
clyro-mcp wrap npx @modelcontextprotocol/server-filesystem /home/user/projects
```

---

## CLI Reference

```
clyro-mcp wrap <server-command> [--config <path>]

Arguments:
  server-command        The full MCP server command and its arguments
  --config, -c <path>   Path to YAML config file
                        (default: ~/.clyro/mcp-wrapper/mcp-config.yaml)
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `CLYRO_API_KEY` | Clyro API key. Required for cloud tracing and policy sync. |
| `CLYRO_API_URL` | Override the backend URL. Default: `https://api.clyro.dev` |

Without `CLYRO_API_KEY` the wrapper runs in local-only mode: governance still applies, audit logs are written locally, but no data is sent to Clyro.

---

## Configuration

The wrapper looks for `~/.clyro/mcp-wrapper/mcp-config.yaml` by default. If the file is absent, the wrapper starts in permissive mode with no policies.

**Minimal config** (just set limits):

```yaml
global:
  max_steps: 100
  max_cost_usd: 20.0
```

**Full example:**

```yaml
global:
  max_steps: 100          # Block after this many tool calls in a session
  max_cost_usd: 20.0      # Block when estimated cost exceeds this (USD)
  cost_per_token_usd: 0.00001

  loop_detection:
    threshold: 3           # Block if the same call repeats ≥ N times
    window: 10             # ...within the last M calls

  policies:                # Global rules — applied to every tool call
    - parameter: "amount"
      operator: "max_value"
      value: 1000
      name: "Max $1000 per transaction"

    - parameter: "destination"
      operator: "not_in_list"
      value: ["external-account-1", "external-account-2"]
      name: "Block external transfers"

tools:
  transfer_funds:          # Per-tool overrides
    policies:
      - parameter: "amount"
        operator: "max_value"
        value: 500
        name: "Stricter limit for fund transfers"

audit:
  log_path: "~/.clyro/mcp-wrapper/mcp-audit.jsonl"
  redact_parameters:       # Patterns to redact from logs (fnmatch syntax)
    - "*password*"
    - "*token*"
    - "*secret*"
    - "*api_key*"

backend:
  agent_name: "my-agent"               # Optional. Default: derived from server command
  sync_interval_seconds: 10            # How often to flush events to cloud (1–300)
  pending_queue_max_mb: 20             # Max disk buffer before dropping events (1–100)
  # api_key: "your-clyro-api-key"            # Prefer CLYRO_API_KEY env var instead
```

### Policy operators

| Operator | Description | `value` type |
|---|---|---|
| `max_value` | Parameter must be ≤ value | number |
| `min_value` | Parameter must be ≥ value | number |
| `equals` | Parameter must equal value | any |
| `not_equals` | Parameter must not equal value | any |
| `in_list` | Parameter must be one of the values | list |
| `not_in_list` | Parameter must not be in the list | list |
| `contains` | Parameter must contain substring | string |
| `not_contains` | Parameter must not contain substring | string |

The `parameter` field supports dot-notation for nested keys (`user.id`) and wildcards (`*.amount`).

---

## Configuration with `--config`

The `--config` (or `-c`) flag lets you pass a per-server YAML config file directly in the launch command. This is useful when you run multiple MCP servers with different governance rules, or when you want to set `agent_name`, `api_key`, and `api_url` per server instead of relying on environment variables or the global default config.

```
clyro-mcp wrap --config /path/to/my_governance.yaml <server-command>
```

### What you can define in the config file

```yaml
global:
  max_steps: 20
  max_cost_usd: 5.0

tools: {}          # Per-tool policy overrides (empty = no per-tool rules)

backend:
  agent_name: "my-agent"          # Name shown in the Clyro dashboard
  api_key: "your-clyro-api-key"         # Clyro API key (overrides CLYRO_API_KEY env var)
  api_url: "https://api.clyro.dev" # Clyro backend URL (overrides CLYRO_API_URL env var)

audit:
  log_path: "/path/to/audit.jsonl"
```

| Field | Description |
|---|---|
| `backend.agent_name` | Identifier for this agent/server in the Clyro dashboard. Defaults to a name derived from the server command. |
| `backend.api_key` | Clyro API key. Overrides the `CLYRO_API_KEY` env var. Prefer the env var for security; use this for per-server keys. |
| `backend.api_url` | Clyro backend URL. Overrides the `CLYRO_API_URL` env var. Useful for pointing at a local dev server (`http://localhost:8000`). |

### Priority order

Config values are resolved in this order (first wins):

1. Explicit `--config` file fields
2. Environment variables (`CLYRO_API_KEY`, `CLYRO_API_URL`)
3. Global default config (`~/.clyro/mcp-wrapper/mcp-config.yaml`)
4. Built-in defaults

### Example: VS Code with `--config`

```json
{
  "mcpServers": {
    "my-governed-server": {
      "type": "stdio",
      "command": "/path/to/venv/bin/clyro-mcp",
      "args": [
        "wrap",
        "--config", "/home/user/configs/my_governance.yaml",
        "/path/to/venv/bin/mcp-server-fetch"
      ],
      "env": {}
    }
  }
}
```

No `CLYRO_API_KEY` env var needed — the `api_key` in the YAML file is used instead.

---

## Wrapping Custom Agents (LangGraph / LangChain)

If you have a custom AI agent that uses MCP tools (e.g., built with LangGraph + `langchain-mcp-adapters`), you can route tool calls through the Clyro wrapper for governance and tracing. The agent launches `clyro-mcp wrap` as the MCP server command instead of the raw server, and the wrapper transparently applies policies.

### How it works

```
[Your Agent]  ←─ stdio ─→  [clyro-mcp wrap]  ←─ stdio ─→  [Your MCP Server]
```

Your agent talks to `clyro-mcp wrap` as if it were the real server. The wrapper enforces governance rules and forwards allowed calls to the actual MCP server.

### Session modes: persistent vs ephemeral

`langchain-mcp-adapters` supports two ways to load tools, and the choice directly affects how the wrapper's subprocess lifecycle works:

| Mode | How tools are loaded | Subprocess lifecycle | Governance session |
|---|---|---|---|
| **Persistent** (recommended) | `client.session()` + `load_mcp_tools(session)` | One subprocess for the entire invocation | Single session — step counts, loop detection, and cost tracking work correctly |
| **Ephemeral** | `client.get_tools()` | New subprocess spawned per tool call | New session per call — step/loop/cost counters reset each time |

Use **persistent sessions** when you need governance to track the full invocation as a single session (step limits, loop detection, cost budgets). Use **ephemeral sessions** when tool calls are independent and you don't need cross-call governance tracking.

### Step 1: Point the client at the wrapper

This is the same regardless of session mode — just change the `command` and `args`:

**Before (direct):**

```python
client = MultiServerMCPClient(
    {
        "my-server": {
            "command": "python",
            "args": ["/path/to/my_mcp_server.py"],
            "transport": "stdio",
        }
    }
)
```

**After (wrapped):**

```python
client = MultiServerMCPClient(
    {
        "my-server": {
            "command": "clyro-mcp",
            "args": [
                "wrap",
                "--config", "/path/to/my_governance.yaml",
                "python", "/path/to/my_mcp_server.py",
            ],
            "transport": "stdio",
        }
    }
)
```

Everything after `wrap` (and the optional `--config <path>`) is treated as the server command. The wrapper starts the server as a subprocess and proxies all MCP communication.

### Step 2: Load tools and build the graph

**Persistent session (recommended):**

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

SERVER_NAME = "my-server"

client = MultiServerMCPClient(
    {
        SERVER_NAME: {
            "command": "clyro-mcp",
            "args": [
                "wrap",
                "--config", "/path/to/my_governance.yaml",
                "python", "/path/to/my_mcp_server.py",
            ],
            "transport": "stdio",
        }
    }
)

async def invoke(user_query: str):
    # All tool calls inside this block share one clyro-mcp subprocess
    async with client.session(SERVER_NAME) as session:
        tools = await load_mcp_tools(session)

        builder = StateGraph(state_schema=MyState)
        builder.add_node("assistant", assistant_node)
        builder.add_node("tools", ToolNode(tools=tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        graph = builder.compile()

        return await graph.ainvoke(
            input={"messages": user_query},
            config={"recursion_limit": 25},
        )
```

The `client.session()` context manager keeps one `clyro-mcp` subprocess alive for the entire invocation. All tool calls go through the same session, so governance counters (step limits, loop detection, cost) accumulate correctly.

**Ephemeral session:**

```python
client = MultiServerMCPClient({SERVER_NAME: { ... }})

async def invoke(user_query: str):
    # Each tool call spawns a new clyro-mcp subprocess
    tools = await client.get_tools()

    builder = StateGraph(state_schema=MyState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(tools=tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    graph = builder.compile()

    return await graph.ainvoke(
        input={"messages": user_query},
        config={"recursion_limit": 25},
    )
```

> **Tip:** The governance config file lets you set `agent_name`, `api_key`, and tool-level policies specific to this agent, keeping each agent's governance independent.

---

## What happens when a tool call is blocked

The wrapper returns a JSON-RPC error to the host instead of forwarding the call to the server. The host sees a clean error response — the server is never invoked. All blocked calls are logged locally and synced to Clyro if an API key is configured.

Blocking reasons:
- **Loop detected** — same tool+args repeated ≥ threshold times in window
- **Step limit** — session has exceeded `max_steps` total tool calls
- **Cost budget** — estimated cost would exceed `max_cost_usd`
- **Policy violation** — a parameter rule matched

---

## Local audit log

Every tool call (allowed and blocked) is appended to:

```
~/.clyro/mcp-wrapper/mcp-audit.jsonl
```

Each line is a JSON event. The log is created with `0600` permissions (owner-readable only). Rotate or truncate it as needed.

---

## Files created by the wrapper

| Path | Description |
|---|---|
| `~/.clyro/mcp-wrapper/mcp-config.yaml` | Your config (you create this) |
| `~/.clyro/mcp-wrapper/mcp-audit.jsonl` | Local audit log |
| `~/.clyro/mcp-agent-<id>.id` | Persisted agent UUID (survives restarts) |
| `~/.clyro/mcp-pending-<id>.jsonl` | Event queue for crash-safe delivery |

---

## Failure behavior

The wrapper is fail-open. If Clyro's backend is unreachable, tool calls are not blocked — they proceed normally. Events are queued to disk and retried on the next sync interval. Local audit logging continues regardless of backend status.

---

## Troubleshooting

### `clyro-mcp: command not found` / `Executable not found in $PATH`

GUI applications (VS Code, Claude Desktop, Cursor) launch processes **without** loading your shell profile (`~/.bashrc`, `~/.zshrc`). Even if `clyro-mcp` works in your terminal, these apps won't find it.

**Fix:** Use the full path to `clyro-mcp` in the app's MCP config. Run `which clyro-mcp` in a terminal where it works to find the path, then replace `"command": "clyro-mcp"` with `"command": "/full/path/to/clyro-mcp"` in the config.

See [Finding the full path to `clyro-mcp`](#finding-the-full-path-to-clyro-mcp) for details.
