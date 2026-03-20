# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-20

### Added

#### SDK Core
- **Local mode**: Full offline operation — no API key or cloud connection required. Set `mode="local"` or omit `api_key` to auto-resolve. Env var `CLYRO_MODE=local|cloud` supported.
- **Local YAML policies**: Declarative policy rules in `~/.clyro/sdk/policies.yaml`. Supports 8 operators (`equals`, `not_equals`, `contains`, `not_contains`, `greater_than`, `less_than`, `max_value`, `min_value`), `block`/`allow`/`require_approval` decisions, and per-rule `fail_open` override.
- **Transport gating**: Zero network calls in local mode — no HTTP client instantiated, no DNS resolution, no connection attempts.
- **Session-end governance summary**: After every run, prints steps, cost, violations, and controls triggered to terminal. Suppressible via `CLYRO_QUIET=1|true`.
- **First-run welcome message**: One-time banner showing SDK version, docs link, and community URL on first `clyro.wrap()` call per process.
- **Policy violation context logging**: On block/require_approval, logs rule name, parameter, operator, expected vs actual values, and CTA to configure team-wide policies.
- **Error context enrichment**: All `ClyroError` subclasses append `Report at github.com/getclyro/clyro/issues` to error messages.
- **`clyro-sdk feedback` CLI**: Submit feedback to cloud API (with API key) or open pre-filled GitHub issue (without). Auto-detects headless environments.

#### Framework Adapters
- **Claude Agent SDK adapter**: Hook-based instrumentation for Anthropic's Claude Agent SDK. Tool correlation, subagent tracking, policy enforcement via `HookRegistrar`.
- **Anthropic SDK adapter**: Proxy-based tracing for `client.messages.create()` and `client.messages.stream()`. Tool use detection, cost tracking, Prevention Stack integration.

#### MCP Governance Wrapper
- **Session-end governance summary**: Prints steps, cost, violations to stderr after MCP session ends. Respects `CLYRO_QUIET`.
- **Error context enrichment**: `ClyroMCPError` messages include issue tracker URL.

#### Claude Code Hooks
- **Core CLI**: PreToolUse/PostToolUse governance for all Claude Code tools (Bash, Edit, Write, MCP). Block/allow decisions via YAML policies.
- **Session state persistence**: Per-session JSON state files with atomic writes and file locking.
- **Cloud policy sync**: Fetch policies from cloud dashboard via `CLYRO_API_KEY`. Same `CloudPolicyFetcher` as MCP wrapper.
- **Trace emission**: PostToolUse events sent to cloud backend for replay and audit.
- **Error context enrichment**: `ClyroHookError` messages include issue tracker URL.

#### Infrastructure
- **Monorepo consolidation**: SDK, MCP wrapper, and Claude Code hooks ship as single `pip install clyro` package with three CLI entry points (`clyro-sdk`, `clyro-mcp`, `clyro-hook`).
- **Shared code deduplication**: Common policy engine, redaction, prevention stack shared across all three components.
- **Apache 2.0 license**: LICENSE file and SPDX headers on all source files.
- **CONTRIBUTING.md**: Development setup, coding conventions, adapter guide.
- **Centralized API URL**: Single `DEFAULT_API_URL` constant in `config.py`.

### Changed
- `ClyroConfig.mode` field added with auto-resolution logic (local if no API key, cloud if API key present).
- `ClyroError.__str__()` now appends issue tracker URL.
- SDK CLI renamed from `clyro` to `clyro-sdk` to avoid collision with monorepo tooling.

### Fixed
- `org_id` resolution no longer fails when API key is absent — gracefully defaults to `"local"`.
- `CLYRO_QUIET` now accepts both `"1"` and `"true"` (case-insensitive).

## [0.1.0] - 2026-02-10

### Added

#### SDK Core
- **`clyro.wrap()` one-line integration**: Decorator/function wrapper for any Python callable (sync or async).
- **Session management**: Automatic session creation, step tracking, duration measurement.
- **Trace event model**: Structured event capture (LLM calls, tool calls, state transitions, errors).
- **HTTP transport**: Background sync to cloud backend with retry (exponential backoff), circuit breaker, and batch uploads.
- **Local SQLite storage**: Offline-first trace buffering with auto-pruning.
- **Execution controls**: Step limits (`StepLimitExceededError`), cost limits (`CostLimitExceededError`), loop detection (`LoopDetectedError`).
- **Cost tracking**: Automatic LLM cost calculation for OpenAI and Anthropic models. Token-based pricing with `tiktoken` integration.
- **Loop detection**: Sliding-window cycle detection with configurable threshold and circuit breaker.
- **Policy enforcement**: Business logic guardrails with 8 operators, Redis caching, fail-closed evaluation.
- **Model selector**: Cost-optimal model recommendations based on task type and budget.
- **Fail-open design**: SDK errors never crash the user's agent.
- **Redaction module**: Automatic PII and secret redaction in logs and traces.

#### Framework Adapters
- **LangGraph adapter**: Node/edge capture, LLM + tool tracing, RAG support via `LangGraphCallbackHandler`.
- **CrewAI adapter**: Task tracing, inter-agent communication, delegation tracking.
- **Generic adapter**: Wraps any Python callable with full Prevention Stack.

#### MCP Governance Wrapper
- **JSON-RPC 2.0 proxy**: stdio transport, tool-call interception.
- **YAML policy config**: Declarative rules with quantity/value/approval enforcement.
- **Loop detection**: Sliding window with configurable threshold.
- **Cost tracking**: Character-count heuristic for MCP tool calls.
- **Audit logger**: JSONL with parameter redaction.

#### Infrastructure
- **OpenTelemetry exporter**: OTLP export from SDK traces.
- **ARI score**: 5-dimension Agent Reliability Index (CII, MCS, AGS, EDS, OCS).
- **Trace hierarchy**: Parent-child event wiring for nested agent calls.

[0.2.0]: https://github.com/getclyro/clyro/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/getclyro/clyro/releases/tag/v0.1.0
