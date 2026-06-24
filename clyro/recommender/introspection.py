# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Policy Recommender — agent introspection
# Implements policy-recommender FRD-PR-001, FRD-PR-002, FRD-PR-004

"""Static reflection of a wrapped agent's shape, across the four frameworks.

Reads an agent's tools, system prompt, model and topology from a *live but
un-run* object — never executing it, never instantiating clients, never invoking
tools. The design is a small **strategy per framework** plus three robustness
layers that real-world agents need:

1. **unwrap** — the object a developer points at is usually a *wrapper instance*
   (their `MyAgent`), not the framework object; the real graph/crew/client lives
   on an attribute (`.graph`, `.agent`, `.client`, `.options`, …). We reach
   through those — including Clyro's own `clyro.wrap()` proxy (`._agent`).
2. **per-framework extractors** — each reads the *verified* attribute paths for
   its framework (e.g. LangGraph tools live in a `ToolNode.tools_by_name`, not a
   top-level `.tools`; CrewAI persona is `role`+`goal`+`backstory`; Claude Agent
   SDK exposes everything on `ClaudeAgentOptions`).
3. **module-scan fallback** — some frameworks keep the system prompt (and, for
   the Anthropic SDK, the tool list) as module-level constants, not on the
   object. We scan the agent's defining module *and the same-package modules it
   imports* for a prompt/tool constant, trying several common names.

Every layer is wrapped so introspection **never raises** (FRD-PR-001/002 failure
clauses) — a weird agent yields a thinner, still-valid result, and the recommender
falls through to its rule-based path.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from clyro.recommender.types import ToolSpec, ToolSurface, TopologyMeta

logger = logging.getLogger("clyro.recommender.introspection")

# Concrete, word-boundary RAG signals — avoids matching the ordinary verb
# "retrieve(s)" or the word "knowledge" in plain prose (which produced false
# "retrieval"/RAG classifications). Matched against tool names + descriptions.
_RAG_HINT_RE = re.compile(
    r"\b(vector(?:store|[\s_-]?db|[\s_-]?search)?|embeddings?|rag|chroma|pinecone|faiss|"
    r"weaviate|qdrant|milvus|pgvector|retriever|knowledge[\s_-]?base|semantic[\s_-]?search)\b",
    re.I,
)

# Attribute / class-name hints that an object wires in MCP servers or clients,
# without connecting to them.
_MCP_ATTRS = ("mcp_servers", "mcp_client", "_mcp_client", "mcp_toolset", "mcp_tools")

# Attributes a wrapper instance commonly uses to hold the real framework object.
_UNWRAP_ATTRS = (
    "graph",
    "_graph",
    "compiled_graph",
    "compiled",
    "app",
    "agent",
    "_agent",
    "crew",
    "client",
    "_client",
    "options",
    "runnable",
)
_UNWRAP_MAX_DEPTH = 6

# Module-level constant names that commonly hold a system prompt (multiple
# variations — different teams name them differently).
_PROMPT_CONST_NAMES = (
    "SYSTEM_PROMPT",
    "SYSTEM_MESSAGE",
    "SYSTEM",
    "SYSTEM_INSTRUCTION",
    "SYSTEM_INSTRUCTIONS",
    "PROMPT",
    "AGENT_PROMPT",
    "ASSISTANT_PROMPT",
    "INSTRUCTIONS",
    "BASE_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
)
# Module-level constant names that commonly hold an Anthropic-style tool list.
_TOOL_CONST_NAMES = (
    "TOOL_SCHEMAS",
    "TOOL_DEFINITIONS",
    "TOOLS",
    "ANTHROPIC_TOOLS",
    "TOOL_SPECS",
    "tool_schemas",
    "tools",
)
_MAX_SCAN_MODULES = 25
_MAX_TOOLS = 200  # cap so a pathological agent can't blow up the fingerprint/prompt


@dataclass
class IntrospectionResult:
    surface: ToolSurface
    system_prompt: str
    model_id: str


@dataclass
class _Partial:
    """What a framework extractor managed to pull out (all optional)."""

    tools: list[ToolSpec] = field(default_factory=list)
    system_prompt: str = ""
    model_id: str = "unknown"
    node_count: int = 0
    multi_agent: bool = False
    has_mcp: bool = False
    has_rag: bool = False


def _safe(fn: Callable[[], Any], default: Any) -> Any:
    """Run ``fn`` and swallow any exception, returning ``default``."""
    try:
        return fn()
    except Exception:
        return default


def _detect_framework(obj: Any) -> str:
    return _safe(lambda: _detect_adapter()(obj), "generic")


def _detect_adapter() -> Callable[[Any], str]:
    from clyro.adapters.generic import detect_adapter

    return detect_adapter


# --- tool coercion (handles all five shapes) ----------------------------------
def _tool_schema(schema: Any) -> dict[str, Any] | None:
    """Coerce a tool's arg schema to a JSON-schema dict (Pydantic v1/v2 or dict)."""
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    for meth in ("model_json_schema", "schema"):  # pydantic v2, then v1
        fn = getattr(schema, meth, None)
        if callable(fn):
            result = _safe(fn, None)
            if isinstance(result, dict):
                return result
    return None


def _coerce_tool(obj: Any) -> ToolSpec | None:
    """Coerce any of the 5 tool shapes into a ToolSpec, or None if not a tool.

    Shapes: langchain StructuredTool / crewai BaseTool (objects with
    ``.name``/``.description``/``.args_schema``); Anthropic dicts
    (``{name, description, input_schema}``); OpenAI dicts (``{function:{name…}}``);
    bare name strings (Claude Agent SDK ``allowed_tools``); plain functions.
    """
    try:
        if isinstance(obj, str):
            name = obj.strip()
            return ToolSpec(name=name) if name else None
        if isinstance(obj, dict):
            fn = obj.get("function") if isinstance(obj.get("function"), dict) else {}
            name = obj.get("name") or fn.get("name")
            if not name:
                return None
            schema = (
                obj.get("input_schema")
                or obj.get("args_schema")
                or obj.get("parameters")
                or fn.get("parameters")
            )
            return ToolSpec(
                name=str(name),
                description=obj.get("description") or fn.get("description"),
                args_schema=_tool_schema(schema),
            )
        name = getattr(obj, "name", None) or getattr(obj, "__name__", None)
        if not name:
            return None
        schema = (
            getattr(obj, "args_schema", None)
            or getattr(obj, "input_schema", None)
            or getattr(obj, "args", None)
        )
        return ToolSpec(
            name=str(name),
            description=getattr(obj, "description", None),
            args_schema=_tool_schema(schema),
        )
    except Exception:
        return None


def _dedupe(specs: list[ToolSpec]) -> list[ToolSpec]:
    out, seen = [], set()
    for s in specs:
        if s and s.name not in seen:
            seen.add(s.name)
            out.append(s)
    return out


def _detect_mcp(obj: Any) -> bool:
    """True if the object exposes MCP wiring (servers/clients) — static, no connect.

    Catches eager MCP usage across frameworks (Claude Agent SDK ``mcp_servers``,
    langchain ``MultiServerMCPClient`` held as an attribute, ``MCPToolset``, …).
    A fully-lazy agent that builds its MCP client only at call time still reads as
    False — there is nothing to see statically.
    """
    try:
        for attr in _MCP_ATTRS:
            if getattr(obj, attr, None):
                return True
        # Inspect *held client objects* (not the agent's own module — a file named
        # ``*_mcp.py`` must not self-trigger) for known MCP client classes.
        for holder in (
            getattr(obj, "client", None),
            getattr(obj, "_client", None),
            getattr(obj, "mcp_client", None),
        ):
            if holder is None:
                continue
            cls = type(holder)
            mod = (getattr(cls, "__module__", "") or "").lower()
            name = (getattr(cls, "__name__", "") or "").lower()
            if (
                mod == "mcp"
                or mod.startswith("mcp.")
                or any(k in mod for k in ("langchain_mcp", "fastmcp", "mcp_use"))
            ):
                return True
            if any(
                k in name
                for k in ("multiservermcpclient", "mcpclient", "clientsession", "mcptoolset")
            ):
                return True
    except Exception:
        return False
    return False


def _runnable_steps(runnable: Any, depth: int = 0):
    """Yield a runnable and the runnables nested inside it (RunnableBinding.bound,
    RunnableSequence.steps) — depth-capped, exception-safe."""
    if runnable is None or depth > 4:
        return
    yield runnable
    bound = _safe(lambda: getattr(runnable, "bound", None), None)
    if bound is not None and bound is not runnable:
        yield from _runnable_steps(bound, depth + 1)
    for step in _safe(lambda: list(getattr(runnable, "steps", []) or []), []):
        yield from _runnable_steps(step, depth + 1)


# --- unwrap -------------------------------------------------------------------
def _unwrap(agent: Any) -> Any:
    """Reach the real framework object through wrapper instances.

    Never instantiates a class; depth-capped; prefers a recognized framework
    object among the holding attributes, else descends into the first holder.
    """
    obj = agent
    seen: set[int] = set()
    for _ in range(_UNWRAP_MAX_DEPTH):
        if _detect_framework(obj) != "generic":
            return obj
        if id(obj) in seen:
            break
        seen.add(id(obj))
        candidates: list[Any] = []
        for attr in _UNWRAP_ATTRS:
            cand = _safe(lambda a=attr, o=obj: getattr(o, a, None), None)
            if cand is not None and cand is not obj and not isinstance(cand, type):
                candidates.append(cand)
        # Prefer a candidate that is itself a recognized framework object.
        for cand in candidates:
            if _detect_framework(cand) != "generic":
                return cand
        if not candidates:
            break
        obj = candidates[0]
    return obj


# --- per-framework extractors -------------------------------------------------
def _extract_claude_agent_sdk(obj: Any) -> _Partial:
    """Claude Agent SDK — ClaudeAgentOptions exposes everything statically.

    Folds the sub-agent roster (``options.agents`` → each ``AgentDefinition`` has
    its own ``prompt`` + tool allow-list) into the surface, not just a count.
    """
    options = getattr(obj, "options", None) or obj
    p = _Partial()

    prompts: list[str] = []
    sp = getattr(options, "system_prompt", None)
    if isinstance(sp, str) and sp.strip():
        prompts.append(sp.strip())
    elif isinstance(sp, dict):
        txt = str(sp.get("preset") or sp.get("text") or "").strip()
        if txt:
            prompts.append(txt)

    model = getattr(options, "model", None) or getattr(options, "fallback_model", None)
    if isinstance(model, str):
        p.model_id = model

    names: list[str] = []
    for attr in ("allowed_tools", "tools"):
        v = getattr(options, attr, None)
        if isinstance(v, (list, tuple)):
            names.extend(str(x) for x in v if isinstance(x, str))

    agents = getattr(options, "agents", None)
    if isinstance(agents, dict):
        p.node_count = len(agents)
        p.multi_agent = len(agents) > 1
        for sub in agents.values():
            st = getattr(sub, "tools", None)
            if isinstance(st, (list, tuple)):
                names.extend(str(x) for x in st if isinstance(x, str))
            sub_p = getattr(sub, "prompt", None)
            if isinstance(sub_p, str) and sub_p.strip():
                prompts.append(sub_p.strip())

    p.tools = _dedupe([ToolSpec(name=n) for n in names])
    p.system_prompt = "\n\n".join(prompts)
    p.has_mcp = bool(getattr(options, "mcp_servers", None))
    return p


def _extract_crewai(obj: Any) -> _Partial:
    """CrewAI — a Crew (has .agents/.tasks) or a single Agent (has .role).

    Includes the hierarchical ``manager_agent`` in the roster and folds
    task-level tools (``task.tools`` can add to / override an agent's tools).
    """
    p = _Partial()
    agents = getattr(obj, "agents", None)
    agent_list = list(agents) if isinstance(agents, (list, tuple)) else [obj]
    manager = getattr(obj, "manager_agent", None)
    if manager is not None and not any(manager is a for a in agent_list):
        agent_list = agent_list + [manager]

    prompts: list[str] = []
    tools: list[ToolSpec] = []
    for a in agent_list:
        for fld in ("role", "goal", "backstory"):
            v = getattr(a, fld, None)
            if isinstance(v, str) and v.strip():
                prompts.append(v.strip())
        atools = getattr(a, "tools", None)
        if isinstance(atools, (list, tuple)):
            tools.extend(s for s in (_coerce_tool(t) for t in atools) if s)
        if p.model_id == "unknown":
            llm = getattr(a, "llm", None)
            m = (
                llm
                if isinstance(llm, str)
                else (getattr(llm, "model", None) or getattr(llm, "model_name", None))
            )
            if isinstance(m, str) and m:
                p.model_id = m

    # Task-level tools (a Task can carry its own tools for that step).
    for task in _safe(lambda: list(getattr(obj, "tasks", []) or []), []):
        ttools = getattr(task, "tools", None)
        if isinstance(ttools, (list, tuple)):
            tools.extend(s for s in (_coerce_tool(t) for t in ttools) if s)

    p.system_prompt = "\n\n".join(prompts)
    p.tools = _dedupe(tools)
    p.node_count = len(agent_list)
    p.multi_agent = len(agent_list) > 1
    return p


def _extract_langgraph(obj: Any) -> _Partial:
    """LangGraph — topology + tools are reachable; prompt/model best-effort.

    Tools: walk ``compiled.nodes`` → each PregelNode's inner runnable (and any
    nested binding/sequence) for a ToolNode ``tools_by_name`` dict OR a
    ``RunnableBinding.kwargs["tools"]`` from ``model.bind_tools([...])``.
    Model/prompt: recovered when the node binds a chat model / ChatPromptTemplate
    as an attribute (e.g. ``create_react_agent``); when they are trapped in a node
    *closure* they stay empty and the module-scan fallback handles the prompt.
    """
    p = _Partial()
    p.tools = _dedupe(_langgraph_tools(obj))
    p.node_count = _langgraph_node_count(obj)
    model_id, prompt = _langgraph_llm_meta(obj)
    if model_id:
        p.model_id = model_id
    if prompt:
        p.system_prompt = prompt
    p.has_mcp = _detect_mcp(obj)
    return p


def _langgraph_tools(graph: Any) -> list[ToolSpec]:
    nodes = _safe(lambda: graph.nodes, None)
    values = list(nodes.values()) if isinstance(nodes, dict) else []
    specs: list[ToolSpec] = []
    for pnode in values:
        inner = getattr(pnode, "node", pnode)
        for r in _runnable_steps(inner):
            tbn = getattr(r, "tools_by_name", None)  # ToolNode
            if isinstance(tbn, dict):
                specs.extend(s for s in (_coerce_tool(t) for t in tbn.values()) if s)
            kw = getattr(r, "kwargs", None)  # RunnableBinding from .bind_tools([...])
            if isinstance(kw, dict) and isinstance(kw.get("tools"), (list, tuple)):
                specs.extend(s for s in (_coerce_tool(t) for t in kw["tools"]) if s)
    return specs


def _langgraph_llm_meta(graph: Any) -> tuple[str, str]:
    """Best-effort (model_id, system_prompt) from a compiled graph's nodes.

    Only succeeds when the model/prompt are reachable as node attributes; returns
    ('', '') otherwise (closures) so the caller falls back to the module scan.
    """
    model_id = ""
    prompt = ""
    for pnode in _safe(lambda: list(graph.nodes.values()), []):
        inner = getattr(pnode, "node", pnode)
        for r in _runnable_steps(inner):
            if not model_id:
                mod = (getattr(type(r), "__module__", "") or "").lower()
                if "chat_models" in mod or "language_models" in mod or "chat_model" in mod:
                    m = getattr(r, "model", None) or getattr(r, "model_name", None)
                    if isinstance(m, str) and m:
                        model_id = m
            if not prompt:
                prompt = _prompt_from_runnable(r)
        if model_id and prompt:
            break
    return model_id, prompt


def _prompt_from_runnable(r: Any) -> str:
    """Pull a system-message string from a ChatPromptTemplate-like runnable."""
    msgs = getattr(r, "messages", None)
    if not isinstance(msgs, (list, tuple)):
        return ""
    for msg in msgs:
        tmpl = _safe(lambda m=msg: getattr(getattr(m, "prompt", None), "template", None), None)
        if isinstance(tmpl, str) and tmpl.strip():
            return tmpl.strip()
        if type(msg).__name__.startswith("System"):
            c = getattr(msg, "content", None)
            if isinstance(c, str) and c.strip():
                return c.strip()
    return ""


def _langgraph_node_count(graph: Any) -> int:
    g = _safe(lambda: graph.get_graph(), None)
    names: list[Any] = []
    if g is not None:
        gn = getattr(g, "nodes", None)
        if isinstance(gn, dict):
            names = list(gn.keys())
        elif gn is not None:
            names = _safe(lambda: list(gn), [])
    if not names:
        nodes = _safe(lambda: graph.nodes, None)
        if isinstance(nodes, dict):
            names = list(nodes.keys())
    return len([n for n in names if n not in ("__start__", "__end__")])


def _extract_anthropic(_obj: Any) -> _Partial:
    """Anthropic SDK — the client holds nothing (tools/prompt/model are per-call).

    Everything comes from the module-scan fallback (the developer's module-level
    SYSTEM_PROMPT + TOOL_SCHEMAS constants).
    """
    return _Partial()


def _extract_generic(obj: Any) -> _Partial:
    """Best-effort attribute scan for frameworks Clyro has no dedicated extractor
    for (OpenAI Agents SDK, AutoGen, LlamaIndex, Pydantic-AI, Google ADK,
    Smolagents). Reads tools as a list *or dict*, a sub-agent roster for topology,
    and a broad set of system-prompt attribute names.
    """
    p = _Partial()

    def _as_items(val: Any) -> list[Any]:
        if callable(val) and not isinstance(val, (list, tuple, dict)):
            val = _safe(val, None)
        if isinstance(val, dict):  # smolagents: {name: Tool}; AutoGen _function_map
            return list(val.values())
        return list(val) if isinstance(val, (list, tuple)) else []

    candidates: list[Any] = []
    for attr in ("tools", "_tools", "toolkit", "_function_map"):
        candidates.extend(_as_items(getattr(obj, attr, None)))

    # Sub-agent rosters → fold member tools + set topology (OpenAI Agents handoffs,
    # ADK sub_agents, AutoGen participants, Smolagents managed_agents).
    members: list[Any] = []
    for attr in ("agents", "sub_agents", "handoffs", "managed_agents", "participants"):
        members.extend(_as_items(getattr(obj, attr, None)))
    for member in members:
        candidates.extend(_as_items(getattr(member, "tools", None)))

    p.tools = _dedupe([s for s in (_coerce_tool(c) for c in candidates) if s])
    if members:
        p.node_count = len(members)
        p.multi_agent = len(members) > 1

    for attr in (
        "system_prompt",
        "system",
        "instructions",
        "instruction",
        "system_message",
        "prompt",
    ):
        v = getattr(obj, attr, None)
        if isinstance(v, str) and v.strip():
            p.system_prompt = v.strip()
            break
    for attr in ("model", "model_name", "_model"):
        v = getattr(obj, attr, None)
        if isinstance(v, str) and v:
            p.model_id = v
            break
    p.has_mcp = _detect_mcp(obj)
    return p


_EXTRACTORS: dict[str, Callable[[Any], _Partial]] = {
    "claude_agent_sdk": _extract_claude_agent_sdk,
    "crewai": _extract_crewai,
    "langgraph": _extract_langgraph,
    "anthropic": _extract_anthropic,
    "generic": _extract_generic,
}


# --- module-scan fallback -----------------------------------------------------
def _candidate_modules(agent: Any) -> list[Any]:
    """The agent's defining module + same-top-package modules it references.

    Covers the common case where the prompt/tools live in a *sibling* module
    (e.g. a LangGraph node module) imported by the agent module.
    """
    mods: list[Any] = []
    seen: set[str] = set()

    def add(modname: str | None) -> None:
        if not modname or modname in seen:
            return
        seen.add(modname)
        m = sys.modules.get(modname)
        if m is not None:
            mods.append(m)

    root_name = getattr(type(agent), "__module__", None)
    add(root_name)
    top = (root_name or "").split(".")[0]
    root_mod = sys.modules.get(root_name or "")
    if root_mod is not None and top:
        for val in _safe(lambda: list(vars(root_mod).values()), [])[:300]:
            if len(mods) >= _MAX_SCAN_MODULES:
                break
            om = getattr(val, "__module__", None)
            if isinstance(om, str) and om.split(".")[0] == top:
                add(om)
    return mods


def _scan_for_prompt(agent: Any) -> str:
    for mod in _candidate_modules(agent):
        for name in _PROMPT_CONST_NAMES:
            val = getattr(mod, name, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def _scan_for_dict_tools(agent: Any) -> list[ToolSpec]:
    for mod in _candidate_modules(agent):
        for name in _TOOL_CONST_NAMES:
            val = getattr(mod, name, None)
            if isinstance(val, (list, tuple)) and val:
                specs = _dedupe([s for s in (_coerce_tool(t) for t in val) if s])
                if specs:
                    return specs
    return []


# --- orchestrator -------------------------------------------------------------
class AgentIntrospector:
    """Pure, defensive static introspection (FRD-PR-001/002/004). Never raises."""

    def introspect(self, agent: Any) -> IntrospectionResult:
        framework = "generic"
        partial = _Partial()
        try:
            target = _unwrap(agent)
            framework = _detect_framework(target)
            extractor = _EXTRACTORS.get(framework, _extract_generic)
            partial = _safe(lambda: extractor(target), _Partial())
        except Exception as exc:  # belt-and-suspenders — must never propagate
            logger.warning("clyro.recommender.introspection_failed: %s", type(exc).__name__)

        # Fallbacks: recover the prompt (LangGraph/Anthropic/generic) and the
        # Anthropic dict-tool list from module-level constants.
        if not partial.system_prompt:
            partial.system_prompt = _safe(lambda: _scan_for_prompt(agent), "")
        if not partial.tools:
            partial.tools = _safe(lambda: _scan_for_dict_tools(agent), [])

        tools = partial.tools[:_MAX_TOOLS]
        blob = " ".join((t.name or "").lower() + " " + (t.description or "").lower() for t in tools)
        topology = TopologyMeta(
            node_count=partial.node_count,
            multi_agent=partial.multi_agent,
            has_rag=partial.has_rag or bool(_RAG_HINT_RE.search(blob)),
            has_mcp=partial.has_mcp or "mcp" in blob or _safe(lambda: _detect_mcp(agent), False),
        )
        surface = ToolSurface(framework=framework, tools=tools, topology=topology)  # type: ignore[arg-type]

        # Visibility into what introspection actually extracted (tools, system
        # prompt, topology, model). Emitted at DEBUG so it is silent in
        # production — enable it with `clyro suggest --debug`, or by raising the
        # "clyro.recommender" logger to DEBUG. Never enable in production.
        if logger.isEnabledFor(logging.DEBUG):
            prompt_preview = " ".join((partial.system_prompt or "").split())[:200]
            logger.debug(
                "introspection result: framework=%s | tools(%d)=%s | "
                "system_prompt(%d chars)=%r | topology=%s | model=%s",
                framework,
                len(tools),
                [t.name for t in tools],
                len(partial.system_prompt or ""),
                prompt_preview,
                topology,
                partial.model_id,
            )

        return IntrospectionResult(
            surface=surface,
            system_prompt=partial.system_prompt,
            model_id=partial.model_id,
        )
