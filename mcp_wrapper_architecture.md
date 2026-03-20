# MCP Wrapper Architecture

**Source**: `clyro/mcp/` (subpackage within the unified `clyro` SDK)

The MCP wrapper is a **governance proxy**. It sits transparently between an AI host
(e.g. Claude Desktop) and a real MCP server. Every tool call the host makes passes through
the wrapper first — the wrapper decides to allow it or block it, logs it, and optionally
syncs the trace to the Clyro cloud backend.

```
[AI Host] ←── stdin/stdout ──→ [clyro.mcp WRAPPER] ←── stdin/stdout ──→ [Real MCP Server]
```

The host and server never know the wrapper exists. They both speak plain newline-delimited
JSON-RPC 2.0 over stdin/stdout as if they were talking directly to each other.

---

## All Files at a Glance

### MCP-specific modules (`clyro/mcp/`)

| File | What it does |
|---|---|
| `__main__.py` | Entry point — just calls `main()` when you run `python -m clyro.mcp` |
| `__init__.py` | Python package marker (empty) |
| `cli.py` | The brain of startup and shutdown. Parses CLI args, orchestrates all component creation, and runs the finally-block cleanup |
| `session.py` | Lightweight state container for one wrapper process — session UUID, step counter, accumulated cost, agent ID |
| `transport.py` | Manages the child MCP server process — spawn, terminate, read/write its stdin/stdout pipes |
| `router.py` | The main runtime loop. Runs 4 async tasks that handle messages flowing between host and server |
| `prevention.py` | Wires the 4 evaluators into a single pipeline. Each call gets an `AllowDecision` or `BlockDecision` |
| `errors.py` | Formats JSON-RPC 2.0 error responses when a call is blocked |
| `audit.py` | Writes every event to a local JSONL audit file AND optionally sends it to the cloud backend. Extends `clyro.audit.BaseAuditLogger` (shared base class) |

### Shared SDK core modules (used by MCP wrapper but live in `clyro/`)

| Module | What it provides | Previously |
|---|---|---|
| `clyro.config` | Pydantic models for every config field + the YAML loader (`WrapperConfig`, `PolicyRule`, `GlobalConfig`, etc.) | `clyro_mcp.config` |
| `clyro.loop_detector` | `LoopDetector`, `compute_call_signature` — detects when the same tool call repeats too many times in a sliding window | `clyro_mcp.loop_detector` |
| `clyro.cost` | `CostTracker`, `HeuristicCostEstimator` — estimates token cost from character count and tracks running total against a budget | `clyro_mcp.cost_tracker` |
| `clyro.policy` | `LocalPolicyEvaluator` — evaluates user-defined rules against tool call parameters | `clyro_mcp.policy_evaluator` |
| `clyro.audit` | `BaseAuditLogger` — shared base class for audit logging across adapters | N/A (new shared base) |
| `clyro.redaction` | `redact_dict_deepcopy` — parameter redaction shared across adapters | N/A (new shared module) |
| `clyro.evaluation` | `enrich_tool_input(use_prefix=False)` — argument enrichment shared across adapters | N/A (new shared module) |

### Backend modules (`clyro/backend/`)

| File | What it does |
|---|---|
| `backend/__init__.py` | Backend subpackage marker (empty) |
| `backend/http_client.py` | Low-level HTTP client (httpx). Handles retries via `_request_with_retry()`, backoff, and auth errors for all API calls |
| `backend/agent_registrar.py` | Registers this wrapper instance as an agent with the cloud. Persists the UUID to disk so it survives restarts |
| `backend/cloud_policy.py` | Fetches policies from the cloud and merges them with locally-configured policies |
| `backend/event_queue.py` | A file-backed JSONL queue. Events land here first, then get flushed to the cloud in batches |
| `backend/circuit_breaker.py` | Protects against a degraded backend — stops sending if failures pile up, retries after a cooldown. Canonical home for `CircuitState`, `ConnectivityStatus`, `CircuitBreakerConfig` types |
| `backend/sync_manager.py` | Runs a background async loop that periodically picks up queued events and sends them to the cloud |
| `backend/trace_event_factory.py` | Builds the structured event dicts that get sent to the cloud (session_start, tool_call, error, etc.) |

---

## Phase 1 — STARTUP

### Step 1 — Entry Point
**`__main__.py`** → **`cli.py`**

`__main__.py` is just two lines — it imports and calls `main()` from `cli.py`. This is the
standard Python idiom that makes `python -m clyro.mcp` work. The `clyro-mcp` CLI command
is also available (entry point in `pyproject.toml`: `clyro.mcp.cli:main`).

`cli.py:_build_parser()` sets up the argument parser. The only supported command is:
```
clyro-mcp wrap <server-command> [--config /path/to/config.yaml]
```

`cli.py:main()` validates that the server command was actually provided, then hands off to
`asyncio.run(_async_main())` — everything from here on is async.

Install: `pip install clyro` (the MCP wrapper is included in the unified SDK package).

---

### Step 2 — Config Loading
**`cli.py:_async_main()`** → **`clyro.config:load_config()`**

The first thing `_async_main` does is load configuration. `clyro.config:load_config()` reads
`~/.clyro/mcp-config.yaml` (or a custom path if `--config` was passed). The file is parsed
with `yaml.safe_load()` — this is intentional, it cannot execute arbitrary Python code.

`clyro.config` defines the full config schema as Pydantic models:

- **`LoopDetectionConfig`** — how many identical calls in a window count as a loop (`threshold`, `window`)
- **`PolicyRule`** — a single rule: which tool parameter to watch, what operator to apply, what value to compare against
- **`ToolConfig`** — a list of `PolicyRule`s scoped to one specific tool by name
- **`AuditConfig`** — where to write the audit log and which parameter names to redact
- **`GlobalConfig`** — session-wide limits: `max_steps`, `max_cost_usd`, `cost_per_token_usd`, loop detection settings, global policies that apply to all tools
- **`BackendConfig`** — API key and URL (both can be overridden via `CLYRO_API_KEY` / `CLYRO_API_URL` env vars), sync toggle, queue size, sync interval
- **`WrapperConfig`** — the root model that combines all of the above

`clyro.config:is_backend_enabled` returns True only if an API key is present AND `sync_enabled`
is not explicitly False. If the config file is missing entirely, `load_config()` returns
a permissive default config (fail-open — the wrapper still runs, just with no limits).

---

### Step 3 — Session Creation
**`cli.py:_async_main()`** → **`session.py`**

`session.py:McpSession` is created next. It is a simple data container scoped to the entire
lifetime of this wrapper process. It holds:
- `session_id` — a fresh UUID4 generated right now
- `step_count` — starts at 0, incremented before every tool call evaluation
- `accumulated_cost_usd` — running total of estimated spend, grows as calls are made
- `agent_id` — initially None, set later when the backend registers this agent

`session.py:PendingCall` is a frozen dataclass also defined here. It stores the details of
a `tools/call` request that has been forwarded to the server but whose response hasn't come
back yet (tool name, params, request ID, timestamp). The router uses it to correlate
responses and calculate the actual cost of each call.

---

### Step 4 — Backend Initialization
**`cli.py:_init_backend()`** → **all `clyro/backend/` files**

This entire step is skipped if no API key is configured. If it is configured, it runs inside
a try/except — any failure here still lets the wrapper start (fail-open).

#### 4A — HTTP Client (`clyro.backend.http_client`)
`HttpSyncClient` is created first. This is the single HTTP client used for all cloud API
calls throughout the session. It stores the API key only in memory (never logs it).
All requests use the shared `_request_with_retry()` method with exponential backoff
(delays of 1s, 2s, 4s) before giving up.

#### 4B — Agent Registration (`clyro.backend.agent_registrar`)
`AgentRegistrar.get_or_register()` ensures this wrapper instance has a stable cloud identity.

`cli.py:_derive_instance_id()` computes a short ID by taking the SHA-256 hash of the agent
name and using the first 12 characters. This deterministic ID is used as a filename suffix
so the same agent always maps to the same on-disk UUID file.

The registrar first checks `~/.clyro/mcp-agent-{instance_id}.id` for a previously saved
UUID. If it exists and is valid, it's reused — no network call needed. If not found,
`clyro.backend.http_client:register_agent()` calls `POST /v1/agents/register` to get a UUID
from the cloud, and `_persist()` saves it to disk with `0o700` directory permissions.
If registration fails (network down, etc.) it falls back to a locally-generated UUID so
the wrapper still starts.

#### 4C — Cloud Policy Fetch (`clyro.backend.cloud_policy`)
`CloudPolicyFetcher.fetch_and_merge()` pulls policy rules from the cloud and merges them
with the user's local config. The fetch has a hard 2-second timeout — if the cloud is slow
or unreachable, it silently falls back to local-only policies (fail-open).

`_extract_rules()` converts the backend's rule format into the same `PolicyRule` shape used
locally. `_merge()` combines them: local rules with the same `name` field override cloud
rules, unsupported operators are skipped with a warning, and `require_approval` actions are
downgraded to `block`.

#### 4D — Event Queue (`clyro.backend.event_queue`)
`EventQueue` is a file-backed JSONL queue stored at `~/.clyro/mcp-pending-{instance_id}.jsonl`.
Every event is written to this file before being sent to the cloud. This means events survive
process crashes — on the next startup, the sync manager will find them and send them.
Max file size is 10MB; if exceeded, the oldest events are pruned. There is also an in-memory
fallback of up to 1000 events if file I/O fails.

#### 4E — Circuit Breaker (`clyro.backend.circuit_breaker`)
`CircuitBreaker` has three states: CLOSED (normal), OPEN (stop sending — backend is down),
HALF_OPEN (probing — try one batch and see if backend has recovered). Thresholds are
hardcoded: 5 consecutive failures trips to OPEN, 30 seconds later it moves to HALF_OPEN,
2 consecutive successes in HALF_OPEN restores to CLOSED.

The canonical types `CircuitState`, `ConnectivityStatus`, and `CircuitBreakerConfig` live
in this module and are shared across all adapters that use the backend.

`ConnectivityDetector` is a simpler companion — it just tracks whether the backend is
CONNECTED or DISCONNECTED and logs a message when the status transitions, so you can see
in logs when the backend goes away or comes back.

#### 4F — Sync Manager + Trace Factory (`clyro.backend.sync_manager`, `clyro.backend.trace_event_factory`)
`TraceEventFactory` is a stateless helper that knows how to build the event dicts the cloud
API expects — it has a method for each event type (session_start, tool_call, blocked_call, etc.).
All events get `framework: "mcp"` and `cost_estimated: true` stamped on them. Output data
is truncated to 10KB to avoid huge payloads.

`BackendSyncManager` is started here. It immediately spawns a background asyncio task
(`_sync_loop`) that will run for the entire life of the wrapper. If the EventQueue has
leftover events from a previous crashed session, an immediate sync is triggered right now.

---

### Step 5 — Core Component Assembly
**`cli.py:_async_main()`**

With config, session, and backend ready, the remaining three core components are assembled:

- **`transport.py:StdioTransport`** — a thin wrapper around `asyncio.create_subprocess_exec()`.
  Not started yet, just constructed.
- **`prevention.py:PreventionStack`** — wires together `clyro.loop_detector.LoopDetector`,
  step limit check, `clyro.cost.CostTracker`, and `clyro.policy.LocalPolicyEvaluator` into
  a single `.evaluate()` call.
- **`audit.py:AuditLogger`** — extends `clyro.audit.BaseAuditLogger`. The audit log file
  (`~/.clyro/mcp-audit.jsonl`) is not opened yet (lazy creation on first write). Permissions
  will be `0o600` — owner read/write only. Parameter redaction uses
  `clyro.redaction.redact_dict_deepcopy` (shared module).

`audit.py:AuditLogger.set_backend()` is then called to attach the `sync_manager` and
`trace_event_factory` to the logger. After this, every `log_tool_call()` and `log_lifecycle()`
call will do two things: write to the local JSONL file AND enqueue a trace event for cloud sync.

---

### Step 6 — Spawn the Real MCP Server
**`transport.py:StdioTransport.start()`**

Now the real MCP server process is spawned. `transport.py:start()` calls
`asyncio.create_subprocess_exec()` with all three streams piped. The child process's stdin
is where the wrapper writes forwarded tool calls. The child's stdout is where responses come
back. The child's stderr is piped so the wrapper can prefix it with `[server]` and forward
it to its own stderr. If the command is not found, the wrapper exits with code 1 immediately.

---

### Step 7 — Session Start + Signal Handlers
**`cli.py:_async_main()`**, **`audit.py:log_lifecycle()`**

`audit.py:log_lifecycle("session_start")` is called. This writes a `session_start` entry to
the local audit JSONL and — since the backend is now attached — enqueues a `session_start`
trace event via `clyro.backend.sync_manager:enqueue()` → `clyro.backend.trace_event_factory:session_start()`
→ `clyro.backend.event_queue:append()`.

Signal handlers are registered:
- SIGTERM and SIGINT → call `router.request_shutdown()` which sets a shutdown event
- SIGHUP → forwarded directly to the child process (lets the server reload config if it supports it)

---

## Phase 2 — RUNTIME

### The Router Loop
**`router.py:MessageRouter.run()`**

This is the steady-state of the wrapper. `MessageRouter.run()` launches exactly 4 concurrent
asyncio tasks and waits for the first one to finish. When any task finishes (or errors), all
others are cancelled and the router exits, triggering the shutdown sequence.

```
Task 1: _host_reader_task()      Reads lines from the AI host's stdin, evaluates each one
Task 2: _server_reader_task()    Reads lines from the server's stdout, correlates costs, forwards to host
Task 3: _stderr_forwarder_task() Reads server stderr, prefixes "[server] ", writes to wrapper stderr
Task 4: _process_monitor_task()  Waits for the child process to exit, logs it, returns exit code
```

---

### When a Tool Call Arrives — The Happy Path (ALLOWED)
**`router.py`** → **`session.py`** → **`prevention.py`** → **`clyro.loop_detector`** → **`clyro.cost`** → **`clyro.policy`** → **`transport.py`** → **`audit.py`** → **`clyro.backend.sync_manager`** → **`clyro.backend.trace_event_factory`** → **`clyro.backend.event_queue`**

`router.py:_host_reader_task()` reads one line at a time from host stdin. Lines larger than
10MB get a warning logged but are forwarded raw. If LSP Content-Length framing is detected
(wrong protocol), a `_FramingError` is raised and the task stops. The line is parsed as JSON.
If it's a JSON-RPC notification (no `id` field) or any method other than `tools/call`, it
passes straight through to the server without evaluation.

For `tools/call`:

1. **`session.py:increment_step()`** — step counter goes up by 1 before any evaluation

2. **`prevention.py:PreventionStack.evaluate()`** — runs 4 stages in order, short-circuits on first violation:

   - **Stage 1 — `clyro.loop_detector:LoopDetector.check()`**: `compute_call_signature()` hashes
     the tool name + params into a SHA-256 digest using canonical JSON (keys sorted, so
     `{"b":1,"a":2}` and `{"a":2,"b":1}` produce the same hash). This signature is appended
     to a fixed-size sliding-window deque. If the same signature appears `threshold` or more
     times within the window, it's a loop — the call is blocked.

   - **Stage 2 — Step limit** (`prevention.py`): If `step_count > max_steps`, the call is blocked.
     Simple integer comparison, no files needed.

   - **Stage 3 — `clyro.cost:CostTracker.check_budget()`**: Estimates cost as
     `(len(params_json) / 4) * cost_per_token_usd * 2`. The 2x multiplier is intentional —
     it accounts for the response we haven't seen yet. Overestimating is acceptable here
     (false positive block) but underestimating is not (we'd exceed budget before catching it).
     If `accumulated_cost + this_estimate > max_cost_usd`, the call is blocked.

   - **Stage 4 — `clyro.policy:LocalPolicyEvaluator.evaluate()`**: Checks per-tool rules
     first (exact tool name match), then global rules. `_resolve_parameter()` extracts the
     relevant value from the tool's params dict using dot-path notation (`"order.quantity"`)
     or wildcard (`"*.amount"`). Argument enrichment uses `clyro.evaluation.enrich_tool_input(use_prefix=False)`
     (shared module). `_evaluate_rule()` applies one of 8 operators:
     `max_value`, `min_value`, `equals`, `not_equals`, `in_list`, `not_in_list`, `contains`,
     `not_contains`. Returns on the first violated rule.

3. **`AllowDecision` returned**:
   - `router.py` stores a `session.py:PendingCall` keyed by request ID (for cost correlation later)
   - `transport.py:write_to_child()` writes the raw original message bytes to the server's stdin
   - `audit.py:log_tool_call(decision="allowed")` runs:
     - `clyro.redaction.redact_dict_deepcopy` makes a deep copy of params and blanks out any keys matching configured fnmatch glob patterns
     - `_write()` appends a JSONL line to the audit file (fail-open: I/O errors are caught and logged, never propagated)
     - `clyro.backend.sync_manager:enqueue()` is called (fail-open):
       `clyro.backend.trace_event_factory:tool_call_act()` builds the event dict →
       `clyro.backend.event_queue:append()` persists it to the JSONL queue file

---

### When a Tool Call Arrives — Blocked Path
**`prevention.py`** → **`errors.py`** → **`router.py`** → **`audit.py`** → **`clyro.backend.sync_manager`** → **`clyro.backend.trace_event_factory`** → **`clyro.backend.event_queue`**

If any prevention stage triggers, a `BlockDecision` is returned with a reason.

`errors.py:format_error()` builds a JSON-RPC 2.0 error response. The error code is always
`-32600` (Invalid Request). The `block_type` field in the error data tells the host exactly
why the call was blocked: `loop_detected`, `step_limit_exceeded`, `budget_exceeded`, or
`policy_violation`. This response is written directly to host stdout by `router.py`.
**The server never sees this call.**

`audit.py:log_tool_call(decision="blocked")` logs it locally and enqueues a `blocked_call()`
trace event (which maps to `event_type="error"` in the backend schema).

---

### When the Server Responds
**`router.py:_server_reader_task()`** → **`clyro.cost`** → **`session.py`** → **`router.py`**

`router.py:_server_reader_task()` reads the server's response. It looks up the matching
`PendingCall` using the JSON-RPC `id` field. With both the original params and the response
content now available, `clyro.cost:CostTracker.accumulate()` computes the actual cost:
`(len(params_json) + len(response_content)) / 4 * cost_per_token_usd`.
`session.py:add_cost()` adds this to the running total. The response is then forwarded
as-is to the host's stdout.

---

### Background Sync Loop (runs the whole time)
**`clyro.backend.sync_manager:_sync_loop()`** → **`clyro.backend.circuit_breaker`** → **`clyro.backend.event_queue`** → **`clyro.backend.http_client`**

Running independently in the background, `_sync_loop` either waits for the configured
interval to pass or gets woken up early by a trigger (e.g. when the queue hits 100 events).

On each iteration:
1. `clyro.backend.circuit_breaker:can_execute()` — if the breaker is OPEN (too many recent
   failures), this sync is skipped entirely. No point hammering a down backend.
2. `clyro.backend.event_queue:load_pending()` loads all queued events. `session_end` and
   `error` events are sorted to the front of the batch (high priority). Up to 100 events
   per batch.
3. `clyro.backend.http_client:send_batch()` sends `POST /v1/traces`. Retries on 429 with the
   `Retry-After` header respected, retries on 5xx with backoff (1s, 2s, 4s). Raises
   `AuthenticationError` on 401/403.
4. On success: `clyro.backend.event_queue:remove_synced()` atomically removes the sent events
   using temp-file + rename. Circuit breaker and connectivity detector are updated.
   If more events remain in the queue, an immediate next sync is triggered.
5. On failure: circuit breaker records the failure. Events stay in the queue for the next
   attempt. If it was an auth error, sync is disabled for the rest of the session.

**Circuit breaker state transitions:**
```
CLOSED ──(5 consecutive failures)──→ OPEN
OPEN   ──(30 seconds pass)        ──→ HALF_OPEN
HALF_OPEN ──(2 successes)         ──→ CLOSED
HALF_OPEN ──(any failure)         ──→ OPEN
```

---

## Phase 3 — SHUTDOWN

Shutdown is triggered by: SIGTERM/SIGINT received, host closes stdin (EOF), server process
exits on its own, or an unhandled exception in any router task.

Everything below runs in the `finally` block of `cli.py:_async_main()`.
**The order here is critical and intentional.**

---

### Shutdown Step 1 — Kill the Server
**`transport.py:StdioTransport.terminate()`**

Sends SIGTERM to the child MCP server process. Waits up to 5 seconds for it to exit cleanly.
If it doesn't, sends SIGKILL. This step happens first so the server is no longer producing
output that could interleave with cleanup.

---

### Shutdown Step 2 — Log Session End  <- MUST happen before Step 3
**`audit.py:AuditLogger.log_lifecycle("session_end")`**

Writes the `session_end` event to the local audit JSONL. Crucially, it also enqueues the
`session_end` trace event via `clyro.backend.sync_manager:enqueue()`. This must happen
**before** the sync manager is shut down in Step 3, so the event is included in the final
flush batch. `clyro.backend.trace_event_factory:session_end()` builds the event, which includes
a `metadata` field with total `step_count` and `accumulated_cost_usd` for the session.

---

### Shutdown Step 3 — Flush the Queue
**`clyro.backend.sync_manager:BackendSyncManager.shutdown()`**

The sync manager does a final flush. It attempts up to 3 batch syncs within a 3-second
total timeout. Any events that still haven't synced after this window are left in the
`clyro.backend.event_queue` file on disk — they will be picked up and sent the next time
the wrapper starts (cross-session recovery). The background `_sync_loop` asyncio task
is then cancelled.

---

### Shutdown Step 4 — Close HTTP Client
**`clyro.backend.http_client:HttpSyncClient.close()`**

The underlying httpx client is closed. No more HTTP calls can be made after this.

---

### Shutdown Step 5 — Close Audit Log
**`audit.py:AuditLogger.close()`**

The JSONL audit file handle is flushed and closed. Any buffered OS writes are flushed to disk.

---

### Shutdown Step 6 — Exit
`router.py` returns an exit code: `0` if the server exited cleanly, `2` if it exited with
a non-zero status, `0` if shutdown was triggered by a signal. `cli.py:main()` calls
`sys.exit()` with this code.

---

## Trace Event Types — What Gets Sent to the Cloud
**`clyro.backend.trace_event_factory`**

| Factory method | `event_type` sent | Triggered when |
|---|---|---|
| `session_start()` | `session_start` | Wrapper finishes startup |
| `session_end()` | `session_end` | Wrapper begins shutdown (includes total steps + cost) |
| `tool_call_act()` | `tool_call` | A tool call is forwarded to the server |
| `tool_call_observe()` | `tool_call` | The server's response is received |
| `blocked_call()` | `error` | A tool call is blocked by any prevention stage |
| `policy_check()` | `policy_check` | Policy evaluation (think stage) |

All events always include `framework: "mcp"` and `cost_estimated: true`.
Output data fields are truncated to 10KB to avoid oversized payloads.

---

## Persistent Files on Disk

| File path | What it stores |
|---|---|
| `~/.clyro/mcp-config.yaml` | User's local config — policies, backend settings, audit settings |
| `~/.clyro/mcp-audit.jsonl` | Append-only local audit log. Every allow/block decision written here. Permissions: `0o600` |
| `~/.clyro/mcp-agent-{instance_id}.id` | The agent's cloud UUID. Persisted so re-registration is avoided across restarts |
| `~/.clyro/mcp-pending-{instance_id}.jsonl` | Event queue for crash recovery. Unsynced events survive process death here |

---

## Key Design Principles

| Principle | What it means in practice |
|---|---|
| **Fail-open everywhere** | Audit write failure, cloud policy fetch failure, backend sync failure — none of these ever block a tool call from being forwarded |
| **Crash-safe event delivery** | Events go to the JSONL queue file before being sent to cloud. A crash mid-session loses nothing |
| **Atomic queue updates** | `clyro.backend.event_queue` uses temp-file + rename so the queue file is never half-written |
| **Dual-mode emission** | Every event goes to both the local audit file AND the cloud backend simultaneously |
| **Safe config parsing** | `yaml.safe_load()` only — no arbitrary Python execution from config files |
| **Secure file permissions** | Audit log `0o600` (owner only). Agent ID directory `0o700` |
| **Cost overestimation is intentional** | The 2x multiplier on pre-call budget checks means the wrapper errs on blocking too early rather than too late |
| **session_end before flush** | Shutdown ordering guarantees the session summary event is always included in the final batch sent to cloud |
| **LSP framing rejected** | If the host sends Content-Length headers (wrong protocol), the wrapper stops with an error rather than silently corrupting the stream |
| **Shared SDK core modules** | Loop detection, cost tracking, policy evaluation, redaction, and config are shared across all adapters (LangGraph, CrewAI, Claude Agent SDK, MCP) via the unified `clyro` package |
