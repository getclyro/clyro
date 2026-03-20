# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Framework Adapters
# Implements PRD-003, PRD-004

"""
Framework adapters for the Clyro SDK.

This module provides adapters for different agent frameworks,
enabling framework-specific tracing and event capture.

Available Adapters:
- GenericAdapter: Basic tracing for any Python callable
- LangGraphAdapter: LangGraph StateGraph tracing with node/state capture
- CrewAIAdapter: CrewAI Crew tracing with task/agent capture
- AnthropicAdapter: Anthropic SDK tracing with LLM/tool event capture
"""

from clyro.adapters.anthropic import (
    AnthropicAdapter,
    AnthropicTracedClient,
    AsyncAnthropicTracedClient,
    is_anthropic_agent,
    validate_anthropic_version,
)
from clyro.adapters.claude_agent_sdk import (
    ClaudeAgentAdapter,
    ClaudeAgentHandler,
    instrument_claude_agent,
    is_claude_agent_sdk_agent,
    validate_claude_agent_sdk_version,
)
from clyro.adapters.crewai import (
    CrewAIAdapter,
    CrewAICallbackHandler,
    is_crewai_agent,
    validate_crewai_version,
)
from clyro.adapters.generic import GenericAdapter, detect_adapter
from clyro.adapters.langgraph import (
    LangGraphAdapter,
    LangGraphCallbackHandler,
    is_langgraph_agent,
    validate_langgraph_version,
)

__all__ = [
    # Generic adapter
    "GenericAdapter",
    "detect_adapter",
    # LangGraph adapter (ADAPTER-001)
    "LangGraphAdapter",
    "LangGraphCallbackHandler",
    "is_langgraph_agent",
    "validate_langgraph_version",
    # CrewAI adapter (ADAPTER-002)
    "CrewAIAdapter",
    "CrewAICallbackHandler",
    "is_crewai_agent",
    "validate_crewai_version",
    # Claude Agent SDK adapter (ADAPTER-003)
    "ClaudeAgentAdapter",
    "ClaudeAgentHandler",
    "instrument_claude_agent",
    "is_claude_agent_sdk_agent",
    "validate_claude_agent_sdk_version",
    # Anthropic SDK adapter (ADAPTER-004)
    "AnthropicAdapter",
    "AnthropicTracedClient",
    "AsyncAnthropicTracedClient",
    "is_anthropic_agent",
    "validate_anthropic_version",
]
