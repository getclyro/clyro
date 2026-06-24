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

Open in wizard: https://app.clyro.dev/agents/new
```

It runs **locally and needs no Clyro account** for the recommendation itself.
Add **`--prefill`** (with an api_key configured) and the CLI POSTs the
recommendation to the backend and prints a one-time `?prefill=<token>` deep-link
the Agent Setup Wizard can consume (see the **`--prefill`** flag below).

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

### Making your package importable (`PYTHONPATH`)

`clyro suggest` runs as a console script, so Python's `sys.path` is the venv's
`bin` directory — **not your current working directory**. If your package isn't
pip-installed, point Python at its root so the import resolves:

```bash
# from anywhere — prepend your project root
PYTHONPATH=/path/to/project clyro suggest yourpkg.module:agent

# or cd into the root and run from there
cd /path/to/project && clyro suggest yourpkg.module:agent
```

For a nested layout, include **every** root your agent imports from
(colon-separated), e.g. a `src/` layout plus a sibling package:

```bash
PYTHONPATH=/repo:/repo/src clyro suggest yourpkg.module:agent
```

(If you get `ModuleNotFoundError: No module named '<yourpkg>'`, this is the fix.)

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
| `--prefill` | POST the recommendation to the backend and print a one-time `?prefill=<token>` wizard link (needs an api_key) |
| `--agent-name <name>` | **re-recommend an existing agent**: derive its `agent_id` as `uuid5(org_id, name)` and tag the prefill. Use the name you govern it under (`config.agent_name` / `clyro.wrap`). |
| `--agent-id <uuid>` | **re-recommend an existing agent** by its exact `agent_id` (overrides `--agent-name`) |
| `--apply` | (preview) route the recommendation to the wizard to apply |
| `-y`, `--yes` | skip the `--apply` confirmation prompt (for non-interactive CI) |
| `--no-cache` | bypass the local fingerprint cache and recompute |
| `--debug` | log what introspection extracted (tools, prompt, topology, model) to stderr — **off by default; never enable in production** |

### `--prefill` — creating the wizard token

Without `--prefill`, the CLI just prints the recommendation; the wizard link is a
plain `…/agents/new`. With `--prefill`, the CLI **sends the recommendation to the
backend** and prints a ready-to-open, pre-filled link:

```bash
clyro suggest myapp:agent --prefill
# → Pre-fill token created. Open in wizard:
#     https://app.clyro.dev/agents/new?prefill=<token>
```

#### Environment variables you must export

`--prefill` reads its credentials from the environment (the CLI calls
`ClyroConfig.from_env()`):

| Variable | Required? | What it does |
|---|---|---|
| `CLYRO_API_KEY` | **Yes** | Authenticates the request. Your `org_id` is **derived from this key**, so there's nothing else to set. |
| `CLYRO_ENDPOINT` | Only when not the default | Backend base URL. Defaults to `https://api.clyro.dev`. **Export it when you target a different environment** (local/staging). |

```bash
export CLYRO_API_KEY="cly_live_…"             # required
export CLYRO_ENDPOINT="http://localhost:8000" # only if not the default api.clyro.dev
clyro suggest myapp:agent --prefill
```

> **`org_id` is automatic** — it's decoded from the api_key, so there is no
> separate org-id variable to export. Just make sure `CLYRO_API_KEY` is the key
> for the org you want to pre-fill into.

> **The `--prefill` route must exist on the backend you point at.** The default
> `https://api.clyro.dev` only works once the policy-recommender endpoints are
> deployed there; until then, set `CLYRO_ENDPOINT` to a backend running the new
> code or you'll get a `404`.

How it works: the CLI resolves your `org_id` from the configured api_key and
`POST`s the payload to
`{endpoint}/v1/organizations/{org_id}/agent-setup/prefill` (header
`X-Clyro-API-Key`). The backend stores it behind a single-use, 10-minute token
and the wizard consumes it on load. If no api_key is configured (or it's a local
key with no org), `--prefill` degrades gracefully — it prints the plain link and
a one-line reason, never erroring out.

### New agent vs. re-recommend an existing one

`--prefill` has **two modes**, chosen only by whether you identify an existing
agent on the command line:

**1. New agent (default)** — the common path: you just built an agent and want to
set up its policies. The wizard **creates the agent** when you finish setup, so
the prefill carries **no `agent_id`**:

```bash
clyro suggest myapp:agent --prefill
# → opens the wizard at /agents/new — Steps 1–5 pre-ticked, agent created on finish
```

**2. Re-recommend an existing agent** — the agent already exists (you set it up
before) and you want to refresh its recommendation after code changes. Identify
the agent so the new recommendation attaches to it:

```bash
# by exact id (copy it from the agent's dashboard page)
clyro suggest myapp:agent --prefill --agent-id 1f2e…-…-…

# …or by the name you govern it under (config.agent_name / clyro.wrap):
#   agent_id is derived as uuid5(org_id, name) — same scheme clyro.wrap() uses,
#   so the two always resolve to the same id.
clyro suggest myapp:agent --prefill --agent-name my-support-agent
```

In re-recommend mode the backend stores the recommendation **against that agent**,
which is what powers the dashboard's **Re-recommend** diff. `--agent-id` wins over
`--agent-name` if you pass both.

> **Picking the mode is the only decision.** Omit both flags for a brand-new
> agent; pass one for an existing agent. If the id you pass doesn't match a
> registered agent, the backend safely ignores it (no error) and treats it like a
> new-agent prefill.

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
  dashboard_base_url: https://app.clyro.dev
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
- **`ModuleNotFoundError: No module named '<yourpkg>'`** → your package root isn't
  on `sys.path` (a console script doesn't add your current directory). Set
  `PYTHONPATH=/path/to/project` (or `cd` into the root) — see
  [Making your package importable](#making-your-package-importable-pythonpath).
- **Other import errors** → run `clyro suggest` in the environment where your
  agent module imports cleanly (required env vars set). The import runs your
  module's top-level code.
- **Thin recommendation** (few tools) → point at the framework object rather than
  a wrapper class, and use `--debug` to see exactly what was extracted.
- **"could not reach the catalogue"** → the first run needs network once to cache
  the catalogue; then offline runs work.
