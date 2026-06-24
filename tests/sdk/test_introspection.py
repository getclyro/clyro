# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Tests for the multi-framework static introspection (policy-recommender
# FRD-PR-001/002/004). Uses duck-typed fakes that match the REAL attribute
# shapes verified against langgraph/crewai/anthropic/claude-agent-sdk.

from __future__ import annotations

import pytest

from clyro.recommender import introspection as I
from clyro.recommender.introspection import (
    AgentIntrospector,
    _coerce_tool,
    _extract_claude_agent_sdk,
    _extract_crewai,
    _extract_langgraph,
    _unwrap,
)

# Module-level constants — exercise the module-scan fallback (Anthropic case),
# which reads SYSTEM_PROMPT / TOOL_SCHEMAS from the agent's defining module.
SYSTEM_PROMPT = "You are a RabbitMQ helper assistant who answers queue questions."
TOOL_SCHEMAS = [
    {"name": "get_total_messages", "description": "count messages",
     "input_schema": {"type": "object", "properties": {"site_name": {}}}},
]


@pytest.fixture
def force_detector(monkeypatch):
    """Route by a `_fw` marker instead of the real framework module check."""
    monkeypatch.setattr(I, "_detect_adapter", lambda: (lambda o: getattr(o, "_fw", "generic")))


# --- tool coercion (the 5 shapes) ---------------------------------------------
class _Schema:
    def model_json_schema(self):  # pydantic v2 shape
        return {"properties": {"site_name": {}}}


class _LCTool:  # langchain StructuredTool / crewai BaseTool shape
    name = "get_total_messages"
    description = "Count messages in a queue"
    args_schema = _Schema()


def _plain_func():
    pass


class TestCoerceTool:
    def test_object_with_pydantic_schema(self):
        spec = _coerce_tool(_LCTool())
        assert spec.name == "get_total_messages"
        assert spec.args_schema == {"properties": {"site_name": {}}}

    def test_anthropic_dict(self):
        spec = _coerce_tool({"name": "x", "description": "d", "input_schema": {"type": "object"}})
        assert spec.name == "x" and spec.args_schema == {"type": "object"}

    def test_openai_function_dict(self):
        spec = _coerce_tool({"function": {"name": "f", "parameters": {"type": "object"}}})
        assert spec.name == "f" and spec.args_schema == {"type": "object"}

    def test_bare_string(self):
        assert _coerce_tool("Bash").name == "Bash"

    def test_plain_function(self):
        assert _coerce_tool(_plain_func).name == "_plain_func"

    def test_non_tool_returns_none(self):
        assert _coerce_tool(42) is None
        assert _coerce_tool({}) is None


# --- unwrap -------------------------------------------------------------------
class _Inner:
    _fw = "langgraph"


class _WrapperWithGraph:  # like the user's RabbitMQAgent (.graph)
    def __init__(self):
        self.graph = _Inner()


class _WrapperWithAgent:  # like RabbitMQCrewAIAgent (.agent)
    def __init__(self):
        self.agent = _Inner()


class TestUnwrap:
    def test_reaches_framework_object_via_graph(self, force_detector):
        assert isinstance(_unwrap(_WrapperWithGraph()), _Inner)

    def test_reaches_via_agent(self, force_detector):
        assert isinstance(_unwrap(_WrapperWithAgent()), _Inner)

    def test_returns_recognized_object_unchanged(self, force_detector):
        inner = _Inner()
        assert _unwrap(inner) is inner

    def test_opaque_object_returned_as_is(self, force_detector):
        obj = object()
        assert _unwrap(obj) is obj


# --- Claude Agent SDK extractor -----------------------------------------------
class _ClaudeOptions:
    system_prompt = "You manage RabbitMQ."
    model = "claude-sonnet-4-5"
    allowed_tools = ["mcp__rabbitmq__get_queues", "Bash"]
    tools = []
    mcp_servers = {"rabbitmq": {"command": "python"}}
    agents = {"helper": object(), "auditor": object()}


def test_extract_claude_agent_sdk():
    p = _extract_claude_agent_sdk(_ClaudeOptions())
    assert p.system_prompt == "You manage RabbitMQ."
    assert p.model_id == "claude-sonnet-4-5"
    assert {t.name for t in p.tools} == {"mcp__rabbitmq__get_queues", "Bash"}
    assert p.has_mcp is True
    assert p.node_count == 2 and p.multi_agent is True


# --- CrewAI extractor ---------------------------------------------------------
class _Llm:
    model = "gpt-4o-mini"


class _CrewAgent:
    role = "RabbitMQ Specialist"
    goal = "Manage queues"
    backstory = "Expert in messaging"
    tools = [_LCTool()]
    llm = _Llm()


class _Crew:
    agents = [_CrewAgent(), _CrewAgent()]


def test_extract_crewai_single_agent():
    p = _extract_crewai(_CrewAgent())
    assert "RabbitMQ Specialist" in p.system_prompt and "Expert in messaging" in p.system_prompt
    assert p.model_id == "gpt-4o-mini"
    assert p.tools[0].name == "get_total_messages"
    assert p.node_count == 1 and p.multi_agent is False


def test_extract_crewai_crew_is_multi_agent():
    p = _extract_crewai(_Crew())
    assert p.node_count == 2 and p.multi_agent is True
    assert len(p.tools) == 1  # deduped across the two identical agents


# --- LangGraph extractor (tools buried in a ToolNode) -------------------------
class _ToolNode:
    def __init__(self, tools):
        self.tools_by_name = {t.name: t for t in tools}


class _PregelNode:
    def __init__(self, node):
        self.node = node


class _DrawGraph:
    def __init__(self, names):
        self.nodes = {n: object() for n in names}


class _CompiledGraph:
    _fw = "langgraph"

    def __init__(self):
        self.nodes = {
            "agent": _PregelNode(object()),
            "tools": _PregelNode(_ToolNode([_LCTool()])),
        }

    def get_graph(self):
        return _DrawGraph(["__start__", "agent", "tools", "__end__"])


def test_extract_langgraph_digs_tools_and_topology():
    p = _extract_langgraph(_CompiledGraph())
    assert [t.name for t in p.tools] == ["get_total_messages"]  # recovered from ToolNode
    assert p.node_count == 2  # __start__/__end__ excluded


# --- orchestration + fallbacks ------------------------------------------------
class _AnthropicClient:
    _fw = "anthropic"  # client is a dead end → must fall back to module-scan


def test_introspect_anthropic_falls_back_to_module_scan(force_detector):
    result = AgentIntrospector().introspect(_AnthropicClient())
    assert result.surface.framework == "anthropic"
    # prompt + dict-tools recovered from this module's SYSTEM_PROMPT / TOOL_SCHEMAS
    assert "RabbitMQ helper assistant" in result.system_prompt
    assert any(t.name == "get_total_messages" for t in result.surface.tools)


def test_introspect_langgraph_end_to_end(force_detector):
    result = AgentIntrospector().introspect(_WrapperWithGraph2())
    assert result.surface.framework == "langgraph"
    assert any(t.name == "get_total_messages" for t in result.surface.tools)


class _WrapperWithGraph2:  # wrapper → compiled graph (unwrap + extract)
    def __init__(self):
        self.graph = _CompiledGraph()


class _Exploding:
    @property
    def tools(self):
        raise RuntimeError("boom")

    @property
    def system_prompt(self):
        raise RuntimeError("boom")


def test_introspect_never_raises_on_hostile_object():
    # No marker → generic extractor; every attribute access raises. The promise
    # is that introspect() degrades safely (no exception) — it returns a valid
    # result with model_id "unknown" (the exploding props couldn't be read).
    # (tools/prompt may be filled by the module-scan fallback from THIS test
    # module's constants — that's expected; the point is nothing propagated.)
    result = AgentIntrospector().introspect(_Exploding())
    assert result.surface.framework == "generic"
    assert result.model_id == "unknown"


# --- proposer robustness (the JSON-parse fix) ---------------------------------
class _FakeTransport:
    name = "fake"

    def __init__(self, replies):
        self._replies = list(replies)

    def invoke(self, prompt):
        return self._replies.pop(0)


def _snapshot():
    from clyro.recommender.catalogue_client import CatalogueSnapshot

    return CatalogueSnapshot(
        agent_type_ids={"agent_type.conversational"},
        concern_ids={"concern.pii-protection"},
        kit_ids={"kit.customer-facing"},
    )


def test_proposer_non_json_raises_validation_error_not_crash():
    from clyro.recommender.proposer import LlmProposer, LlmValidationError
    from clyro.recommender.types import ToolSurface

    surface = ToolSurface(framework="generic")
    # garbage both times → LlmValidationError (NOT a raw JSONDecodeError)
    proposer = LlmProposer(_FakeTransport(["not json at all", "still not json"]), _snapshot())
    with pytest.raises(LlmValidationError):
        proposer.propose(surface, "")


def test_proposer_recovers_on_retry():
    from clyro.recommender.proposer import LlmProposer
    from clyro.recommender.types import ToolSurface

    good = ('{"detected_agent_type":"agent_type.conversational","alternative_agent_types":[],'
            '"recommended_kits":[],"inferred_concerns":[{"id":"concern.pii-protection"}]}')
    proposer = LlmProposer(_FakeTransport(["garbage", good]), _snapshot())
    out = proposer.propose(ToolSurface(framework="generic"), "")
    assert out["detected_agent_type"] == "agent_type.conversational"
