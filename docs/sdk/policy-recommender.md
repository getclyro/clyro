# Clyro SDK — Policy Recommender (`clyro suggest`)

Point Clyro at an agent you've **already built** and it tells you what to govern:
the agent's **type**, the **concerns** worth tracking, and the **kits** (concern
bundles) to apply — each with a rationale and a confidence score. It reads your
agent's tools, system prompt, and structure; it never runs the agent.

```bash
clyro suggest myapp.agents:support_agent
```
```
Detected agent type: agent_type.transactional
  or: agent_type.retrieval

Recommended kits:
  • kit.regulated-starter (best-fit) [medium] — Covers 33% of the inferred concerns…

Inferred concerns:
  • concern.pii-protection      [high]   — Tool argument `email` is PII.
  • concern.reversibility       [high]   — Tool `refund_customer` performs an irreversible action.
  • concern.approval-gates      [medium] — Tool `refund_customer` warrants an approval gate.

Open in wizard: https://clyro.dev/agents/new
```

It runs **locally and needs no Clyro account** for the recommendation itself.
(The wizard deep-link will carry a one-time `?prefill=<token>` that auto-fills the
form once the Agent Setup Wizard ships; today it opens the wizard.)

---

## Install

```bash
pip install clyro
```

Requires Python ≥ 3.11. The command is available as both `clyro` and `clyro-sdk`.

---

## Quick start

`clyro suggest` takes a **Python import path** to your agent object —
`module.path:object` (or `module.path.object`):

```bash
# a module-level agent object
clyro suggest myapp.agents.support:agent

# a LangGraph compiled graph
clyro suggest myapp.graphs:compiled_graph

# a CrewAI Agent or Crew
clyro suggest myapp.crew:research_crew
```

`clyro suggest` **imports your module** to get the object, so run it in the same
environment your agent runs in (the one where `import myapp.agents.support`
works, with any required env vars set). It does **not** execute the agent or call
its tools.

---

## What it produces

| Field | Meaning |
|---|---|
| `detected_agent_type` | one of 6 archetypes (conversational, transactional, decisioning, retrieval, code-assistant, workflow-automation) |
| `alternative_agent_types` | runner-up types (shown when the detection is close) |
| `inferred_concerns` | risks worth governing (PII, reversibility, approval-gates, cost, tool-scope, …) each with a rationale + confidence |
| `recommended_kits` | curated concern bundles that fit the agent |
| `sector_hint` | a soft BFSI/Pharma/Retail hint from the prompt (optional) |
| `transport_used` | which LLM path ran (or `rule-based`) |

Use `--json` to get the machine-readable payload (same shape the Agent Setup
Wizard consumes).

---

## Flags

| Flag | Effect |
|---|---|
| `--llm-transport <auto\|claude-code\|anthropic-api\|rule-based>` | which LLM (if any) refines the recommendation. Default: your config, else `auto`. |
| `--json` | print the JSON payload to stdout (suppresses colour) |
| `--out <file>` | write the JSON payload to a file |
| `--apply` | (preview) route the recommendation to the wizard to apply |
| `-y`, `--yes` | skip the `--apply` confirmation prompt (for non-interactive CI) |
| `--no-cache` | bypass the local fingerprint cache and recompute |
| `--debug` | log what introspection extracted (tools, prompt, topology, model) to stderr — **off by default; never enable in production** |

---

## Transports — rule-based by default, LLM optional

The recommendation always has a deterministic **rule-based backbone**. An LLM can
*refine* it (better nuance + rationale), but it is held on a tight leash: it can
only choose from the real catalogue — it can never invent a concern or kit.

| `--llm-transport` | Behaviour |
|---|---|
| `auto` *(default)* | try Claude Code CLI → Anthropic API key → rule-based; first available wins |
| `claude-code` | use the `claude` CLI (no separate key needed). Errors loudly if `claude` isn't installed |
| `anthropic-api` | use `ANTHROPIC_API_KEY`. Errors loudly if no key is set |
| `rule-based` | skip the LLM entirely — deterministic; ideal for CI |

```bash
clyro suggest myapp:agent --llm-transport rule-based   # fully offline, deterministic
```

Configure a default in `clyro.config.yaml`:
```yaml
policy_recommender:
  llm_transport: auto
  dashboard_base_url: https://clyro.dev
```

---

## How it works (in 4 steps)

1. **Introspect** — read the agent's tools (names + arg schemas), system prompt,
   topology (nodes/agents, RAG, MCP), and model. *Never runs the agent.*
2. **Map** — deterministic rules turn that shape into catalogue ids
   (e.g. a `refund` tool → *transactional* + *reversibility*/*approval-gates*;
   an `email` argument → *PII protection*).
3. **(Optional) refine with an LLM** — schema-gated to the catalogue.
4. **Emit** — a recommendation, cached by a fingerprint of the agent so re-runs
   on unchanged code are instant.

---

## Framework support

`clyro suggest` introspects all four supported frameworks. How much it can read
**statically** (without running the agent) varies by how each framework exposes
its internals:

| Framework | Tools | System prompt | Model | Topology |
|---|---|---|---|---|
| **Claude Agent SDK** | ✅ allowed tools | ✅ `system_prompt` | ✅ | ✅ subagents + MCP |
| **CrewAI** | ✅ | ✅ role/goal/backstory | ✅ | ✅ agents/tasks |
| **LangGraph** | ✅ (from tool nodes) | ⚠️ via module-scan¹ | ⚠️ best-effort | ✅ nodes |
| **Anthropic SDK** | ⚠️ via module-scan¹ | ⚠️ via module-scan¹ | ⚠️ | — |

¹ Some frameworks keep the prompt/tool list as **module-level constants** (e.g.
`SYSTEM_PROMPT`, `TOOL_SCHEMAS`) rather than on the object. Clyro scans your
agent's module (and the sibling modules it imports) for these. If your prompt is
a local variable inside a function, it can't be read statically — the
recommendation still works from the tools + topology, just with less nuance.

**Pointing at the right object** — if your agent is wrapped in a class, point at
the framework object (Clyro also unwraps common holders automatically):

```bash
clyro suggest myapp.rmq:RabbitMQAgent          # may be thin if tools are buried
# better — point at the framework object:
python -c "from myapp.rmq import RabbitMQAgent; RabbitMQAgent()"   # see .graph/.agent/.client
```
A wrapped `clyro.wrap(agent)` object is also unwrapped automatically.

Introspection **never raises** — an exotic or dynamically-built agent yields a
thinner, still-valid recommendation rather than an error.

---

## Offline & the catalogue

To map your agent to ids, the SDK fetches the public catalogue once
(`GET /v1/agent-types`, `/concerns`, `/kits` — no api_key) and caches a snapshot
at `~/.clyro/catalogue-snapshot.json`. After the first run it works fully
offline. The recommendation cache lives at `~/.clyro/proposer-cache.db`.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `2` | bad import path / invalid flag / unwritable `--out` |
| `3` | explicit transport unavailable (e.g. `--llm-transport claude-code` with no `claude`) |
| `4` | explicit transport failed at runtime |
| `5` | unexpected failure |

---

## Troubleshooting

- **"Rule-based only…"** in the header → no LLM transport was available; install
  Claude Code or set `ANTHROPIC_API_KEY`, or pass `--llm-transport rule-based` to
  silence it.
- **Import error** → run `clyro suggest` in the environment where your agent
  module imports cleanly (required env vars set). The import runs your module's
  top-level code.
- **Thin recommendation** (few tools) → point at the framework object rather than
  a wrapper class, and use `--debug` to see exactly what was extracted.
- **"could not reach the catalogue"** → the first run needs network once to cache
  the catalogue; then offline runs work.
