# Clyro Claude Code Hooks — Usage Guide

A governance layer for Claude Code (Anthropic's CLI). It hooks into Claude Code's tool-call lifecycle to enforce policies, track costs, detect loops, and sync traces to Clyro — without modifying Claude Code itself.

```
[Claude Code]
    ↓ (stdin JSON)
[clyro-hook evaluate]       ← PreToolUse: 4-stage prevention stack
    ↓ (allow / block)
[Tool call proceeds or is blocked]
    ↓
[clyro-hook trace --event tool-complete]   ← PostToolUse: cost correction, trace
    ↓
[clyro-hook trace --event session-end]     ← Stop: session summary, event flush
```

Each hook invocation is an **ephemeral process** — no background daemons or long-lived connections. All state is persisted to disk as JSON files.

---

## Install

```bash
pip install clyro
```

This installs the unified Clyro SDK which includes the `clyro-hook` CLI command.

**For local development** (editable install from source):

```bash
pip install -e /path/to/clyro/code/backend/sdk
```

Requires Python ≥ 3.11.

---

## Integration

### Step 1: Create a policy config

Copy the example config to the default location:

```bash
mkdir -p ~/.clyro/hooks
cp claude-code-policy.example.yaml ~/.clyro/hooks/claude-code-policy.yaml
```

Or start with a minimal config:

```yaml
global:
  max_steps: 50
  max_cost_usd: 10.0
```

### Step 2: Register hooks with Claude Code

Add the hooks to your Claude Code settings file (`~/.claude/settings.json`).

**Minimal setup** (govern all tools):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "clyro-hook evaluate --config ~/.clyro/hooks/claude-code-policy.yaml"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "clyro-hook trace --event tool-complete --config ~/.clyro/hooks/claude-code-policy.yaml"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "clyro-hook trace --event session-end --config ~/.clyro/hooks/claude-code-policy.yaml"
          }
        ]
      }
    ]
  }
}
```

**With existing hooks** (e.g., alongside a files blocker and prompt injection defender):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read|Edit|Write|Bash|Grep",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude/hooks/files_blocker.py"
          }
        ]
      },
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "clyro-hook evaluate --config ~/.clyro/hooks/claude-code-policy.yaml"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Read|WebFetch|Task|Bash|Grep",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude/hooks/prompt-injection-defender/post-tool-defender.py"
          }
        ]
      },
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "clyro-hook trace --event tool-complete --config ~/.clyro/hooks/claude-code-policy.yaml"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "clyro-hook trace --event session-end --config ~/.clyro/hooks/claude-code-policy.yaml"
          }
        ]
      }
    ]
  }
}
```

That's it. Claude Code will now invoke `clyro-hook` on every tool call.

### Hook configuration format

Each hook entry in the settings file uses this structure:

| Field | Description | Required |
|---|---|---|
| `matcher` | Regex pattern matched against the tool name (e.g., `".*"` for all tools, `"Bash\|Edit"` for specific tools). Not needed for `Stop` hooks. | No (defaults to all) |
| `hooks` | Array of hook commands to execute | Yes |
| `hooks[].type` | Must be `"command"` | Yes |
| `hooks[].command` | Shell command to run | Yes |

Multiple hook entries in the same lifecycle phase run in order. If an earlier hook blocks the tool call (e.g., `files_blocker.py`), later hooks in the same phase do not run.

### How it works

- **PreToolUse** (`clyro-hook evaluate`): Claude Code pipes a JSON payload to stdin. The hook runs a 4-stage prevention stack and either outputs nothing (allow) or a JSON block decision to stdout.
- **PostToolUse** (`clyro-hook trace --event tool-complete`): Corrects pre-call cost estimates with actuals, emits trace events, writes audit log.
- **Stop** (`clyro-hook trace --event session-end`): Computes session summary, flushes queued events to the Clyro backend, cleans up stale sessions.

---

## What Gets Governed

| Check | When | Default |
|---|---|---|
| Loop detection | Same tool+args repeated ≥ N times in last M calls | 3 repeats in 10 calls |
| Step limit | Total tool calls in session exceeds limit | 50 steps |
| Cost budget | Estimated accumulated cost exceeds budget | $10.00 |
| Policy rules | Tool parameters match a block rule | No rules (permissive) |

All four checks run on every `PreToolUse` invocation, in that order. The first check that triggers a block stops evaluation — the tool call is rejected and Claude Code sees the reason.

---

## What Gets Traced

| Event | When |
|---|---|
| `session_start` | First tool call of a session (or after a turn restart) |
| `pre_tool_use` | Every PreToolUse evaluation (allow or block) |
| `policy_check` | Governance decision with individual rule results |
| `tool_call_observe` | PostToolUse — actual cost, duration, output summary |
| `error` | When a tool call is blocked (linked to parent policy_check) |
| `session_end` | Stop hook — session totals (steps, cost, duration) |

Events are queued to disk during the session and flushed in a single batch at session-end.

---

## Configuration

The hook looks for `~/.clyro/hooks/claude-code-policy.yaml` by default. Pass `--config <path>` to use a different file. If the file is absent, the hook starts in **permissive mode** with no policies.

### Minimal config

```yaml
global:
  max_steps: 50
  max_cost_usd: 10.0
```

### Full example

```yaml
# ── Global limits ───────────────────────────────────────────────────────
global:
  max_steps: 50              # Max tool calls per session before blocking
  max_cost_usd: 10.0         # Max estimated token cost ($USD) per session
  cost_per_token_usd: 0.00001  # Cost heuristic: chars/4 * this value

  loop_detection:
    threshold: 3             # Block if same tool+args repeated this many times
    window: 10               # Within the most recent N tool calls

  # Global policies — evaluated for ALL tools
  policies:
    - name: "Block recursive force delete"
      parameter: command
      operator: not_contains
      value: "rm -rf"

    - name: "Block /etc writes"
      parameter: command
      operator: not_contains
      value: "/etc/"

# ── Per-tool policies ─────────────────────────────────────────────────
tools:
  Bash:
    policies:
      - name: "Block sudo commands"
        parameter: command
        operator: not_contains
        value: "sudo"

      - name: "Block curl to external"
        parameter: command
        operator: not_contains
        value: "curl"

# ── Cloud backend (optional) ─────────────────────────────────────────
backend:
  api_key: null              # Or set CLYRO_API_KEY env var
  api_url: "https://api.clyro.dev"
  agent_name: "claude-code-session"
  policy_cache_ttl_seconds: 300  # How often to refresh cloud policies

# ── Audit logging ────────────────────────────────────────────────────
audit:
  log_path: "~/.clyro/hooks/audit.jsonl"
  redact_parameters:
    - "*password*"
    - "*token*"
    - "*secret*"
    - "*api_key*"
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

### Synthetic parameters

In addition to the tool's own parameters, the hook injects synthetic `_clyro_*` parameters that you can reference in policy rules:

| Parameter | Value |
|---|---|
| `_clyro_tool_name` | The tool being called (e.g., `Bash`, `Edit`, `Write`) |
| `_clyro_session_id` | Current session identifier |
| `_clyro_step_number` | Current step count in the session |
| `_clyro_cost` | Accumulated cost estimate so far (USD) |
| `_clyro_agent_id` | Cloud agent identity (if registered) |

**Example:** Block all tool calls after step 25 specifically for the `Write` tool:

```yaml
tools:
  Write:
    policies:
      - name: "Limit Write tool to first 25 steps"
        parameter: _clyro_step_number
        operator: max_value
        value: 25
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `CLYRO_API_KEY` | Clyro API key. Required for cloud tracing and policy sync. |
| `CLYRO_API_URL` | Override the backend URL. Default: `https://api.clyro.dev` |

Environment variables override values in the config file.

Without `CLYRO_API_KEY` the hook runs in **local-only mode**: governance still applies, audit logs are written locally, but no data is sent to Clyro.

---

## CLI Reference

```
clyro-hook evaluate [--config <path>]
```

Reads a tool call from stdin, runs the 4-stage prevention stack, and outputs a block decision to stdout (or nothing to allow). Exit code 0 = decision rendered. Exit code 1 = internal error, fail-open.

```
clyro-hook trace --event <tool-complete|session-end> [--config <path>]
```

Handles post-tool-use cost correction and session-end lifecycle. Always exits 0.

| Argument | Description |
|---|---|
| `evaluate` | Run the prevention stack (PreToolUse hook) |
| `trace --event tool-complete` | Record tool result and correct cost (PostToolUse hook) |
| `trace --event session-end` | Flush events, write summary, clean up (Stop hook) |
| `--config, -c <path>` | Path to YAML config file. Default: `~/.clyro/hooks/claude-code-policy.yaml` |

---

## The 4-Stage Prevention Stack

Every `PreToolUse` invocation runs these checks in order. The first failing check blocks the tool call.

### Stage 1: Loop Detection

Hashes `(tool_name, tool_input)` and checks if the same hash appears ≥ `threshold` times within the last `window` calls.

**Block reason:** `"Loop detected: same tool call repeated N times (threshold: M)"`

### Stage 2: Step Limit

Checks if the next step would exceed `max_steps`.

**Block reason:** `"Step limit exceeded: N > M"`

### Stage 3: Cost Budget

Estimates cost as `(len(params_json) / 4) * cost_per_token_usd * 2` (the 2x multiplier accounts for expected output). Checks if accumulated cost plus the estimate would exceed `max_cost_usd`.

> **Note:** The pre-call estimate intentionally overestimates. The actual cost is corrected in PostToolUse using the real response size.

**Block reason:** `"Cost budget exceeded: $X.XX accumulated would exceed $Y.YY"`

### Stage 4: Policy Rules

Evaluates local YAML policies merged with cloud policies (if configured). Tool input parameters are enriched with `_clyro_*` synthetic parameters before evaluation.

**Block reason:** `"Policy violation: {rule_name} ({param} {op} {expected}, actual: {actual})"`

---

## Cloud Backend Integration

When `CLYRO_API_KEY` is set, the hook connects to the Clyro backend for:

- **Agent registration** — Registers the hook as an agent in the Clyro dashboard. Falls back to a deterministic UUID if the backend is unreachable.
- **Cloud policy sync** — Fetches policies defined in the Clyro dashboard and merges them with local YAML policies. Cached with a configurable TTL (default: 300 seconds).
- **Trace event delivery** — Events queued during the session are flushed in a single batch at session-end.
- **Violation reporting** — Blocked tool calls are reported with full context (rule name, operator, expected vs actual values).

All backend calls are protected by a **circuit breaker** (5 failures → open, 30s cooldown → half-open, 2 successes → closed). The circuit breaker state is persisted in the session file.

### Priority order for configuration

Config values are resolved in this order (first wins):

1. Environment variables (`CLYRO_API_KEY`, `CLYRO_API_URL`)
2. Config file fields (`backend.api_key`, `backend.api_url`)
3. Built-in defaults

---

## Local Audit Log

Every tool call (allowed and blocked) is appended to:

```
~/.clyro/hooks/audit.jsonl
```

Each line is a JSON event. The log is created with `0600` permissions (owner-readable only). Rotate or truncate it as needed.

### Parameter redaction

Sensitive parameters are redacted before writing to the audit log. The default patterns (fnmatch glob syntax) are:

- `*password*`
- `*token*`
- `*secret*`
- `*api_key*`

Redaction applies recursively to nested dictionaries and lists.

---

## Files Created by the Hook

| Path | Description |
|---|---|
| `~/.clyro/hooks/claude-code-policy.yaml` | Your config (you create this) |
| `~/.clyro/hooks/audit.jsonl` | Local audit log (append-only JSONL) |
| `~/.clyro/hooks/sessions/{session_id}.json` | Session state (step count, cost, loop history, circuit breaker) |
| `~/.clyro/hooks/sessions/{session_id}.lock` | File lock for concurrent access |
| `~/.clyro/hooks/agents/hook-agent-{id}.id` | Persisted cloud agent UUID |
| `~/.clyro/hooks/pending/pending-{session_id}.jsonl` | Event queue (flushed at session-end) |

Session files older than 24 hours are automatically cleaned up at session-end.

---

## Common Policy Examples

### Block dangerous shell commands

```yaml
global:
  policies:
    - name: "Block rm -rf"
      parameter: command
      operator: not_contains
      value: "rm -rf"

    - name: "Block system shutdown"
      parameter: command
      operator: not_contains
      value: "shutdown"

tools:
  Bash:
    policies:
      - name: "Block sudo"
        parameter: command
        operator: not_contains
        value: "sudo"

      - name: "Block package installs"
        parameter: command
        operator: not_contains
        value: "pip install"
```

### Restrict file writes to specific directories

```yaml
tools:
  Write:
    policies:
      - name: "Block writes outside project"
        parameter: file_path
        operator: contains
        value: "/home/user/my-project"
```

### Set strict limits for expensive sessions

```yaml
global:
  max_steps: 25
  max_cost_usd: 5.0
  loop_detection:
    threshold: 2
    window: 5
```

---

## Failure Behavior

The hook is **fail-open**. If the config file is missing, session state is corrupted, the cloud backend is unreachable, or any internal error occurs — the tool call is **allowed to proceed**. Errors are logged to stderr (visible in Claude Code's hook output) and exit code 1 is returned, which Claude Code treats as a non-blocking result.

This means:
- A misconfigured policy file never breaks your Claude Code workflow.
- Backend outages don't block local development.
- Corrupt state files are automatically reset.

Events that couldn't be flushed remain in the pending queue file and are recovered on the next session-end.
