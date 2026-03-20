# Claude Code Hooks Architecture

**Source**: `clyro/hooks/` (subpackage within the unified `clyro` SDK)

The `clyro.hooks` package is a **governance CLI tool** invoked by Claude Code's hook system. Every
tool call Claude Code makes (Bash, Edit, Write, Read, MCP, etc.) passes through the hook as a
subprocess — the hook decides to allow it or block it, logs it, and optionally queues trace events
for the Clyro cloud backend.

```
                    PreToolUse                      PostToolUse / Stop
[Claude Code] ───── stdin JSON ────→ [clyro-hook evaluate] ────→ stdout JSON (block) or empty (allow)
              ───── stdin JSON ────→ [clyro-hook trace --event tool-complete]
              ───── stdin JSON ────→ [clyro-hook trace --event session-end]
```

**Key difference from the MCP wrapper**: The MCP wrapper is a long-running proxy process. `clyro-hook`
is **ephemeral** — one short-lived process per hook event. All state must be persisted to disk between
invocations, and there is no background sync loop. Trace events are queued to a file and flushed in
a single batch at session-end.

**Installation**: `pip install clyro` (the hooks are included in the unified SDK package).

---

## All Files at a Glance

| File | What it does | Shared SDK core? |
|---|---|---|
| `__init__.py` | Package marker. Stores `__version__ = "0.1.0"`. Documents FRD-HK-001 through FRD-HK-012 | No |
| `cli.py` | Entry point (`clyro.hooks.cli:main`). Two subcommands: `evaluate` (PreToolUse) and `trace` (PostToolUse/Stop). Reads stdin JSON, orchestrates all components, writes stdout | No |
| `config.py` | Pydantic models + YAML loader. Extends `clyro.config.WrapperConfig` with hook-specific defaults. Env var overrides | Extends shared |
| `constants.py` | All default paths, exit codes, timeouts, and tuning constants in one place. Defaults derived from `clyro.config.GlobalConfig` and `clyro.config.LoopDetectionConfig` field defaults | Derives from shared |
| `models.py` | Pydantic data models: `HookInput`, `HookOutput`, `SessionState`, `CircuitBreakerSnapshot`, `PolicyCache` | No |
| `state.py` | File-based session state persistence. Atomic writes (temp + rename), `fcntl` file locking, path traversal sanitization, stale session cleanup | No |
| `evaluator.py` | The brain: 4-stage prevention stack (loop → step → cost → policy). Argument enrichment delegates to `clyro.evaluation.enrich_tool_input(use_prefix=True)`. Emits trace events for allow/block decisions | Uses shared |
| `tracer.py` | PostToolUse cost adjustment and trace event emission. Session-end summary, event queue flush, stale cleanup. Token estimation delegates to `clyro.cost.HeuristicCostEstimator` | Uses shared |
| `backend.py` | Agent registration (imports from `clyro.backend.agent_registrar`), circuit breaker (imports from `clyro.backend.circuit_breaker`), event queue (file + memory fallback), trace event factory (output truncation deduplicated: `create_trace_event` calls `truncate_output`), violation reporting | Uses shared |
| `audit.py` | Append-only JSONL audit logger extending `clyro.audit.BaseAuditLogger`. Parameter redaction uses `clyro.redaction.redact_params`. Fail-open on I/O errors | Extends shared |
| `policy_loader.py` | Fetches cloud policies, merges with local YAML, manages TTL cache in session state | Uses shared |

---

## Phase 1 — INVOCATION

### How Claude Code Invokes the Hook

Claude Code's `.claude/settings.json` (or global settings) registers the hook:

```json
{
  "hooks": {
    "PreToolUse": [{ "command": "clyro-hook evaluate" }],
    "PostToolUse": [{ "command": "clyro-hook trace --event tool-complete" }],
    "Stop": [{ "command": "clyro-hook trace --event session-end" }]
  }
}
```

On every tool call, Claude Code spawns `clyro-hook` as a subprocess, writes a JSON payload to
its stdin, and reads stdout for the decision. The process exits immediately after — there is no
long-lived connection.

The `clyro-hook` CLI command is registered as an entry point: `clyro.hooks.cli:main`.

---

### Step 1 — Entry Point
**`cli.py:main()`**

`main()` builds an `argparse` parser with two subcommands:

```
clyro-hook evaluate [--config /path/to/config.yaml]
clyro-hook trace --event {tool-complete,session-end} [--config /path/to/config.yaml]
```

The `evaluate` command routes to `cmd_evaluate()`. The `trace` command routes to `cmd_trace()`.

---

### Step 2 — Read stdin
**`cli.py:_read_stdin()`** → **`cli.py:_parse_hook_input()`**

The hook reads all of stdin as raw text and parses it as JSON. The JSON is validated into a
`HookInput` Pydantic model:

- **`HookInput`** (`models.py`):
  - `session_id: str` — required, identifies the Claude Code session
  - `tool_name: str` — which tool (Bash, Edit, Write, Read, etc.)
  - `tool_input: dict` — the tool's parameters (e.g. `{"command": "ls -la"}`)
  - `tool_result: dict | None` — only present in PostToolUse (the tool's output)

If stdin is empty, JSON is malformed, or `session_id` is missing, the hook exits with code 1
(fail-open — Claude Code allows the tool call).

---

### Step 3 — Config Loading
**`cli.py:cmd_evaluate()`** → **`config.py:load_hook_config()`**

`load_hook_config()` reads `~/.clyro/hooks/claude-code-policy.yaml` (or a custom path via
`--config`). The file is parsed with `yaml.safe_load()` — this is intentional, it cannot execute
arbitrary Python code.

**`config.py:HookConfig`** extends `clyro.config.WrapperConfig` (reusing the same schema so
`LocalPolicyEvaluator`, `CostTracker`, etc. work without adaptation). It adds one hook-specific field:

- `policy_cache_ttl_seconds: int = 300` — how often to refresh cloud policies

The config schema hierarchy:
- **`GlobalConfig`** (`clyro.config`) — `max_steps`, `max_cost_usd`, `cost_per_token_usd`, `loop_detection` (threshold + window), global `policies`
- **`ToolConfig`** — per-tool policy rules (e.g. Bash-specific rules)
- **`BackendConfig`** — `api_key`, `api_url`, `agent_name`
- **`AuditConfig`** — `log_path`, `redact_parameters` (glob patterns)

If the config file is missing, `_hook_defaults()` returns permissive defaults (fail-open). Env var
overrides: `CLYRO_API_KEY` overrides `backend.api_key`, `CLYRO_API_URL` overrides `backend.api_url`.

---

### Step 4 — Agent ID Resolution
**`cli.py:_ensure_agent_id()`** → **`backend.py:resolve_agent_id()`**

Agent ID functions are imported from `clyro.backend.agent_registrar` (shared SDK module, not
reimplemented locally).

On the first invocation of a session, `resolve_agent_id()` obtains a cloud identity for this
hook instance. This is needed for cloud policy fetching and trace event attribution.

The resolution follows a 3-step waterfall:

1. **State check**: If `SessionState.agent_id` is already set (from a prior invocation), reuse it.
2. **Persisted file**: Check `~/.clyro/hooks/agents/hook-agent-{instance_id}.id` for a previously
   saved UUID. The `instance_id` is a deterministic SHA-256 hash (first 12 chars) of the agent
   name, so the same agent always maps to the same file.
   - If the file exists and is **confirmed** (no `.unconfirmed` marker), use it directly.
   - If **unconfirmed**, try re-registering with the backend.
3. **Backend registration**: `POST /v1/agents/register` via `HttpSyncClient` (`clyro.backend.http_client`)
   to get a UUID from the cloud. If this fails, generate a local UUID and mark it unconfirmed.

The agent_id is persisted in `SessionState` so subsequent invocations skip this step entirely.

---

## Phase 2a — PreToolUse (EVALUATE)

### The Prevention Stack
**`cli.py:cmd_evaluate()`** → **`evaluator.py:evaluate()`**

This is the critical path. Every PreToolUse hook invocation runs the full 4-stage prevention
stack. The stack short-circuits on the first violation — later stages are not evaluated.

```
┌─────────────────────────────────────────────────────────────────┐
│                 evaluate(hook_input, config, audit)              │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────┐   ┌───────┐│
│  │ Stage 1      │   │ Stage 2      │   │ Stage 3  │   │Stage 4││
│  │ Loop Detect  │──→│ Step Limit   │──→│ Cost     │──→│Policy ││
│  │ (FRD-HK-003) │   │ (FRD-HK-004) │   │(FRD-005) │   │(006)  ││
│  └──────┬───────┘   └──────┬───────┘   └────┬─────┘   └───┬───┘│
│         │ block?           │ block?         │ block?       │    │
│         ↓                  ↓                ↓              ↓    │
│  _save_and_block    _save_and_block   _save_and_block   report │
│  _emit_block_trace  _emit_block_trace _emit_block_trace  + emit│
└─────────────────────────────────────────────────────────────────┘
```

---

### Stage 1 — Loop Detection (FRD-HK-003)
**`evaluator.py`** → **`clyro.loop_detector.LoopDetector`**

`LoopDetector` is imported from the `clyro` SDK core. It uses a sliding-window deque to
detect when the same tool call (name + params hashed to SHA-256) repeats too many times.

The hook reconstructs the deque from `SessionState.loop_history` (a list of hash strings persisted
across invocations) and calls `loop_detector.check(tool_name, tool_input)`. Default thresholds
are derived from `clyro.config.LoopDetectionConfig` field defaults:
3 repetitions within a window of 10 calls.

If triggered → block with reason `"Loop detected: same tool call repeated N times"`.

---

### Stage 2 — Step Limit (FRD-HK-004)
**`evaluator.py`**

Simple integer comparison: `next_step = state.step_count + 1`. If `next_step > config.max_steps`,
the call is blocked. Default limit derived from `clyro.config.GlobalConfig`: 50 steps per session.

---

### Stage 3 — Cost Budget (FRD-HK-005)
**`evaluator.py`** → **`clyro.cost.CostTracker`**

`CostTracker` is imported from the `clyro` SDK core. Token estimation delegates to
`clyro.cost.HeuristicCostEstimator`. It estimates the cost of a tool call as:
`(len(params_json) / 4) * cost_per_token_usd * 2`. The 2x multiplier accounts for the unseen
response. If `accumulated_cost + estimate > max_cost_usd`, the call is blocked.
Default budget: $10.00 per session.

---

### Stage 4 — Policy Rules (FRD-HK-006)
**`evaluator.py`** → **`policy_loader.py`** → **`clyro.policy.LocalPolicyEvaluator`**

Before evaluation, tool input is **enriched** using the shared SDK module
`clyro.evaluation.enrich_tool_input(use_prefix=True)`:

```python
enriched["_clyro_tool_name"] = tool_name
enriched["_clyro_session_id"] = session_id
enriched["_clyro_step_number"] = step_count
enriched["_clyro_cost"] = accumulated_cost_usd
enriched["_clyro_agent_id"] = agent_id  # if present
```

This enables meta-rules like `_clyro_step_number max_value 100` or `_clyro_agent_id equals my-agent`.

**Policy merging** (`policy_loader.py:get_merged_policies()`):
1. Collect all local policies (global + per-tool from YAML)
2. If API key configured and agent_id available:
   - Check `SessionState.policy_cache` for cached cloud policies (TTL: 300s default)
   - If cache stale, fetch from cloud via `CloudPolicyFetcher.fetch_and_merge()` (`clyro.backend.cloud_policy`)
   - Cloud fetch is protected by circuit breaker (skip if breaker open)
   - Fail-open: any error falls back to local-only policies
3. Merged policies are cached in `SessionState.policy_cache` for subsequent invocations

`LocalPolicyEvaluator.evaluate()` (from `clyro.policy`) checks the enriched input against all merged rules.
It supports 8 operators: `max_value`, `min_value`, `equals`, `not_equals`, `in_list`, `not_in_list`,
`contains`, `not_contains`. Returns on the first violated rule.

---

### Allow Decision (All Stages Pass)

When all 4 stages pass:

1. **Cost estimation**: Pre-call cost estimate is computed and stored in `SessionState.pre_call_cost_estimate`
   (will be corrected in PostToolUse when actual response length is known)
2. **State update**: `step_count` incremented, `loop_history` updated, `accumulated_cost_usd` adjusted
3. **State persistence**: `save_state()` writes atomically (temp file + rename)
4. **Audit log**: `audit.log_pre_tool_use(decision="allow")` appends a JSONL line
5. **Trace events** (FRD-HK-008):
   - On first step (`step_count == 1`): emit `session_start` trace event
   - Always: emit `policy_check` trace event with `decision: "allow"` and all `rule_results`
6. **stdout**: Empty (no output) — Claude Code interprets empty stdout as "allow"
7. **Exit code**: 0

---

### Block Decision (Any Stage Triggers)

When any stage triggers:

1. **State persistence**: `_save_and_block()` saves loop history and timestamps
2. **Audit log**: `audit.log_pre_tool_use(decision="block", reason=...)` with rule results
3. **Violation report** (Stage 4 only, FRD-HK-006): If API key and agent_id configured,
   `backend.py:report_violation()` sends a rich payload to the cloud:
   - `policy_id`, `rule_id`, `rule_name`, `operator`, `expected_value`, `actual_value`
   - `parameters_hash` (SHA-256 of canonical tool_input JSON)
   - Protected by circuit breaker (fail-open)
4. **Trace events** (FRD-HK-008):
   - `policy_check` trace event with `decision: "block"` — gets a UUID `event_id`
   - `error` trace event with `parent_event_id` wired to the policy_check event_id
   - `error_type` maps to the block type: `loop_detected`, `step_limit_exceeded`,
     `budget_exceeded`, or `policy_violation`
5. **stdout**: JSON object `{"decision": "block", "reason": "..."}`
6. **Exit code**: 0 (decision was rendered successfully)

---

### State Locking
**`cli.py:cmd_evaluate()`** → **`state.py:StateLock`**

The entire `evaluate()` call runs inside a `StateLock` context manager. This uses `fcntl.flock()`
with `LOCK_EX` (exclusive) and a 5-second timeout to prevent race conditions when Claude Code
fires concurrent hook invocations for the same session.

If the lock cannot be acquired within 5 seconds, the hook exits with code 1 (fail-open).

---

### Latency Guard
**`cli.py:cmd_evaluate()`**

After evaluation completes, elapsed time is checked. If > 200ms, a warning is logged. PreToolUse
hooks are on the critical path — every millisecond of latency delays the user's tool call.

---

## Phase 2b — PostToolUse (TRACE)

### Tool Call Complete
**`cli.py:cmd_trace()`** → **`tracer.py:handle_tool_complete()`**

This hook fires after a tool call returns. It cannot block — exit code is always 0.

1. **Cost correction** (FRD-HK-008): The pre-call cost estimate (computed in PreToolUse) is
   replaced with actual cost computed from `len(params_json) + len(response_json)`.
   Token estimation delegates to `clyro.cost.HeuristicCostEstimator`:
   ```
   accumulated_cost = accumulated_cost - pre_call_estimate + actual_cost
   pre_call_cost_estimate = 0.0
   ```

2. **Duration calculation**: `duration_ms = now - state.last_tool_call_at` (the timestamp set
   during PreToolUse evaluation)

3. **State persistence**: Updated cost and cleared pre-call estimate are saved atomically
   (within a `StateLock` to prevent race with concurrent PreToolUse)

4. **Audit log**: `audit.log_post_tool_use()` writes a `tool_call_observe` JSONL entry

5. **Trace event** (FRD-HK-008): If API key configured, a `tool_call_observe` event is created
   via `create_trace_event()` with:
   - UUID `event_id`
   - `input_data` (redacted tool input) and `output_data` (result summary — metadata only,
     not raw content: e.g. `"stdout: 1234 chars, exitCode: 0"`)
   - `token_count_input` and `token_count_output` (estimated via `clyro.cost.HeuristicCostEstimator`)
   - `duration_ms`, `step_number`, `accumulated_cost_usd`
   - The event is enqueued to the file-based event queue (not sent immediately)

### Latency Guard

PostToolUse has a 500ms latency threshold (more generous than PreToolUse's 200ms since it's
not on the critical path).

---

## Phase 3 — SHUTDOWN (Stop Hook)

### Session End
**`cli.py:cmd_trace()`** → **`tracer.py:handle_session_end()`**

This hook fires once when the Claude Code session ends.

1. **Session summary**: Computes total `duration_seconds` from `state.started_at` to now

2. **Audit log**: `audit.log_session_end()` writes a summary JSONL entry with
   `total_steps`, `total_cost_usd`, `duration_seconds`

3. **Trace event** (FRD-HK-009): If API key configured, a `session_end` event is created
   via `create_trace_event()` with metadata containing session totals

4. **Event queue flush** (FRD-HK-009): `flush_event_queue()` sends all queued trace events
   (accumulated during the entire session) to the backend in a single batch:
   - Protected by circuit breaker (skip if breaker open)
   - `HttpSyncClient.send_batch()` (`clyro.backend.http_client`) sends `POST /v1/traces`
   - On success: queue file is deleted, circuit breaker records success
   - On failure: events stay in queue file for next session's startup
     (cross-session recovery via file persistence)

5. **Stale session cleanup** (FRD-HK-002): `cleanup_stale_sessions()` deletes session state
   files older than 24 hours. Only runs at session-end to avoid overhead on every PreToolUse.

---

## Backend Integration

### Trace Event Factory
**`backend.py:create_trace_event()`**

All trace events are created through a single factory function that produces SDK-compatible dicts.
Output truncation is deduplicated internally — `create_trace_event` calls `truncate_output`:

```python
{
    "event_id": "uuid4",           # Unique per event — enables parent wiring
    "event_type": "...",           # session_start, policy_check, tool_call_observe, error, session_end
    "session_id": "...",
    "agent_id": "...",
    "agent_name": "...",
    "parent_event_id": "..." | None,  # Wires error events to their policy_check parent
    "tool_name": "...",
    "framework": "claude-code-hooks",
    "timestamp": "ISO-8601",
    "duration_ms": 0,
    "input_data": {...},           # Redacted tool input
    "output_data": {...},          # Truncated to 10KB max
    "token_count_input": 0,        # Estimated via clyro.cost.HeuristicCostEstimator
    "token_count_output": 0,
    "cost_usd": 0.0,
    "step_number": 0,
    "accumulated_cost_usd": 0.0,
    "error_type": "..." | None,
    "error_message": "..." | None,
    "metadata": {
        "_source": "claude-code-hooks",
        "cost_estimated": true,
        ...
    }
}
```

Output data is truncated to 10KB (`OUTPUT_TRUNCATE_BYTES`) to avoid oversized payloads —
matching the MCP wrapper's behavior.

---

### Event Queue (File-Based JSONL + Memory Fallback)
**`backend.py:enqueue_event()`**, **`backend.py:load_queued_events()`**, **`backend.py:clear_event_queue()`**

Events are appended to `~/.clyro/hooks/pending/pending-{session_id}.jsonl` one line at a time.
The file uses `O_APPEND` mode so concurrent invocations can safely write without corruption.

If file I/O fails (disk full, permission denied), events fall back to an in-memory dict
(`_memory_fallback`) capped at 1000 events per session. Since each hook invocation is a separate
process, this fallback primarily helps when the disk issue is transient within a single invocation.

`load_queued_events()` combines events from both the file and memory fallback.
`clear_event_queue()` removes both the file and any memory fallback entries.

---

### Circuit Breaker (Ephemeral-Adapted)
**`backend.py`** → imports `check_can_execute`, `record_success`, `record_failure` from **`clyro.backend.circuit_breaker`**

The circuit breaker protects against hammering a degraded backend. The core logic is provided by
the shared SDK module `clyro.backend.circuit_breaker` — the hooks package imports the functions
rather than reimplementing them locally.

Since the hook is ephemeral, the breaker state is persisted in `SessionState.circuit_breaker`
(a `CircuitBreakerSnapshot` Pydantic model) and loaded/saved with every invocation.

State transitions (matching MCP wrapper):
```
CLOSED ──(5 consecutive failures)──→ OPEN
OPEN   ──(30 seconds pass)        ──→ HALF_OPEN
HALF_OPEN ──(2 successes)         ──→ CLOSED
HALF_OPEN ──(any failure)         ──→ OPEN
```

---

### Violation Reporting
**`backend.py:report_violation()`**

Policy violations (Stage 4 blocks) are reported to the cloud with a rich payload for governance
analytics:

| Field | Source |
|---|---|
| `agent_id`, `session_id` | Session state |
| `action_type` | Tool name |
| `policy_id` | From `violation_details` |
| `rule_id`, `rule_name` | From `violation_details` |
| `operator` | e.g. `not_contains` |
| `expected_value`, `actual_value` | JSON-serialized |
| `parameters_hash` | SHA-256 of canonical `tool_input` JSON |
| `step_number` | Current step count |
| `decision` | Always `"block"` |
| `timestamp` | UTC ISO-8601 |

Protected by circuit breaker. Fail-open on any error.

---

## Trace Event Types — What Gets Queued for the Cloud

**`backend.py:create_trace_event()`** + **`evaluator.py`** + **`tracer.py`**

| Event type | Triggered when | Emitted by |
|---|---|---|
| `session_start` | First tool call of the session | `evaluator.py:_emit_allow_trace_events()` |
| `policy_check` (allow) | Every tool call that passes all 4 stages | `evaluator.py:_emit_allow_trace_events()` |
| `policy_check` (block) | Every tool call blocked by any stage | `evaluator.py:_emit_block_trace_events()` |
| `error` | Every blocked call (child of policy_check) | `evaluator.py:_emit_block_trace_events()` |
| `tool_call_observe` | PostToolUse — tool response received | `tracer.py:handle_tool_complete()` |
| `session_end` | Stop hook — session ending | `tracer.py:handle_session_end()` |

All events include `framework: "claude-code-hooks"`, `cost_estimated: true`, and `_source: "claude-code-hooks"`.

---

## Audit Logging
**`audit.py:AuditLogger`** (extends `clyro.audit.BaseAuditLogger`)

Every decision is written to a local JSONL file at `~/.clyro/hooks/audit.jsonl` (configurable).
The audit log is completely independent of the cloud backend — it works even with no API key.

**Events logged:**
- `pre_tool_use` — every PreToolUse evaluation (allow or block), with redacted tool input
- `tool_call_observe` — every PostToolUse trace, with duration_ms
- `session_end` — session summary with total steps, cost, duration

**Parameter redaction** (delegates to `clyro.redaction.redact_params`):
- Recursive into nested dicts and lists
- Uses fnmatch glob patterns (default: `*password*`, `*token*`, `*secret*`, `*api_key*`)
- If redaction fails for a specific value, it's replaced with `[REDACTION_ERROR]`
- Redaction applies to both audit log and trace events

**File handling:**
- Lazy open (file created on first write, not on construction)
- `O_APPEND` mode with `0o600` permissions (owner read/write only)
- Fail-open: I/O errors are logged but never propagated

---

## Session State Management
**`state.py`**

All state is persisted as JSON files at `~/.clyro/hooks/sessions/{session_id}.json`.

**`SessionState`** fields:
| Field | Purpose |
|---|---|
| `session_id` | Session identifier |
| `agent_id` | Cloud agent UUID (resolved once, reused) |
| `step_count` | Number of tool calls made |
| `accumulated_cost_usd` | Running cost total |
| `loop_history` | List of call signature hashes for loop detection |
| `started_at` | Session start timestamp |
| `last_tool_call_at` | Last PreToolUse timestamp (for duration_ms) |
| `policy_cache` | Cached cloud policies with TTL |
| `cloud_disabled` | Set to true on auth errors (stops cloud calls) |
| `pre_call_cost_estimate` | PreToolUse estimate, corrected in PostToolUse |
| `circuit_breaker` | Persisted breaker state (state, failure_count, opened_at, etc.) |

**Atomic writes** (`state.py:save_state()`):
Write to `.tmp` file first, then `os.rename()` to the final path. This prevents half-written
state files from corrupting the session.

**File locking** (`state.py:StateLock`):
Uses `fcntl.flock(LOCK_EX)` with a 5-second timeout on a `.lock` file adjacent to the state
file. Prevents concurrent PreToolUse/PostToolUse invocations from corrupting shared state.

**Path traversal protection** (`state.py:state_path()`):
Session IDs are sanitized — path separators and `..` are stripped, and only alphanumeric +
`._-` characters are allowed. Invalid IDs map to `"invalid-session"`.

**Stale cleanup** (`state.py:cleanup_stale_sessions()`):
Session files older than 24 hours are deleted. Only runs during the Stop hook (session-end)
to avoid overhead on every PreToolUse.

---

## Persistent Files on Disk

| File path | What it stores |
|---|---|
| `~/.clyro/hooks/claude-code-policy.yaml` | User's local config — policies, backend settings, audit settings |
| `~/.clyro/hooks/audit.jsonl` | Append-only local audit log. Every allow/block decision. Permissions: `0o600` |
| `~/.clyro/hooks/sessions/{session_id}.json` | Session state — step count, cost, loop history, circuit breaker, policy cache |
| `~/.clyro/hooks/sessions/{session_id}.lock` | fcntl lock file for concurrent access protection |
| `~/.clyro/hooks/agents/hook-agent-{instance_id}.id` | Cloud agent UUID. Persisted to avoid re-registration |
| `~/.clyro/hooks/pending/pending-{session_id}.jsonl` | Event queue for trace events. Flushed at session-end. Survives crashes |

---

## Monorepo SDK Integration

`clyro.hooks` is a subpackage within the unified `clyro` SDK. Shared functionality is imported
from SDK core modules rather than duplicated locally. This table shows every shared dependency:

| Component | Imported from | Adaptation in hooks |
|---|---|---|
| `LoopDetector` | `clyro.loop_detector` | Loop history reconstructed from persisted state each invocation |
| `CostTracker` | `clyro.cost` | Same API, but cost accumulated across ephemeral processes via state file |
| `HeuristicCostEstimator` | `clyro.cost` | Token estimation for trace events and cost correction |
| `LocalPolicyEvaluator` | `clyro.policy` | Same API, same 8 operators |
| `WrapperConfig` | `clyro.config` | Extended as `HookConfig` with hook-specific defaults |
| `GlobalConfig`, `LoopDetectionConfig` | `clyro.config` | Default constants derived from field defaults |
| `PolicyRule`, etc. | `clyro.config` | Reused directly |
| `CloudPolicyFetcher` | `clyro.backend.cloud_policy` | Same fetch + merge logic |
| `HttpSyncClient` | `clyro.backend.http_client` | Same HTTP client for all backend calls |
| `resolve_agent_id`, etc. | `clyro.backend.agent_registrar` | Imported directly (not reimplemented locally) |
| `check_can_execute`, `record_success`, `record_failure` | `clyro.backend.circuit_breaker` | Imported directly; state persisted in `SessionState` for ephemeral model |
| `BaseAuditLogger` | `clyro.audit` | Extended with hook-specific log methods |
| `redact_params` | `clyro.redaction` | Used for both audit log and trace event redaction |
| `enrich_tool_input` | `clyro.evaluation` | Called with `use_prefix=True` to add `_clyro_*` parameters |

---

## Key Design Principles

| Principle | What it means in practice |
|---|---|
| **Ephemeral process model** | One short-lived process per hook event. All state persisted to disk. No background tasks or long-lived connections |
| **Fail-open everywhere** | Config missing, state corrupt, cloud unreachable, audit write failure — none of these ever block a tool call |
| **Crash-safe state** | Atomic writes (temp + rename). fcntl locks prevent concurrent corruption. File-based event queue survives process crashes |
| **Latency-conscious** | PreToolUse warns at 200ms. No unnecessary I/O on the critical path. Cloud policy fetch cached with TTL |
| **MCP wrapper parity** | Same prevention stack, same trace event format, same violation payload. Cloud backend cannot distinguish hook events from MCP wrapper events |
| **Monorepo code sharing** | Shared modules (loop detection, cost tracking, policy evaluation, redaction, circuit breaker, agent registration) live in SDK core and are imported — not copied. This ensures behavioral consistency across hooks and MCP wrapper |
| **Secure file permissions** | Session state `0o600`. Agent ID directory `0o700`. Audit log `0o600` |
| **Safe config parsing** | `yaml.safe_load()` only — no arbitrary Python execution from config files |
| **Parent event wiring** | Block decisions produce two events: `policy_check` (parent) → `error` (child), linked by `event_id` / `parent_event_id` |
| **Cost overestimation is intentional** | The 2x multiplier on pre-call budget checks means the hook errs on blocking too early rather than too late |
| **session_end before flush** | The session_end trace event is enqueued before the event queue is flushed, ensuring it's always included in the final batch |
| **Path traversal prevention** | Session IDs are sanitized to prevent writing state files outside the sessions directory |

---

## Exit Code Contract

| Exit code | Meaning | Used by |
|---|---|---|
| `0` | Decision rendered (PreToolUse) or trace completed (PostToolUse/Stop) | All successful invocations |
| `1` | Internal error — fail-open. Claude Code allows the tool call | Invalid input, config error, lock timeout, unexpected exception |
| `2` | Never used by clyro-hook. Claude Code treats this as a blocking error | Reserved — intentionally avoided |
