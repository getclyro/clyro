# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""Anthropic adapter local-mode policy tests — FRD-020 (A5), A10.

Regression: ``AnthropicAdapter.create_traced_client`` gated the policy evaluator
on ``api_key``::

    if self._config.controls.enable_policy_enforcement and self._config.api_key:

``wrap()`` returns this adapter's traced client directly (wrapper.py) rather than
a WrappedAgent, so the wrapper's own SDKLocalPolicyEvaluator is never built for
this path. In local mode there is no api_key, so the evaluator was ``None`` and
policies in ~/.clyro/sdk/policies.yaml were silently ignored — no enforcement,
and no dry-run would_block either.

Step and cost limits masked it: they are session-level controls that need no
evaluator, so they kept working and made the gap look dry-run specific.

OpenAI already had ``_build_policy_evaluator``; these tests assert the two
adapters agree, since ``wrap()`` dispatches to them through the same branch.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from clyro.adapters.anthropic import AnthropicAdapter
from clyro.adapters.openai import OpenAIAdapter
from clyro.config import ClyroConfig, ExecutionControls
from clyro.local_policy import SDKLocalPolicyEvaluator
from clyro.policy import PolicyEvaluator

_CLOUD_KEY = "cly_test_dummy_key_for_tests"


def _adapter(cls, *, api_key=None, enforcement_mode="enforce", policies=True):
    cfg = ClyroConfig(
        agent_name="test-agent",
        api_key=api_key,
        controls=ExecutionControls(
            enable_policy_enforcement=policies,
            enforcement_mode=enforcement_mode,
        ),
    )
    a = cls.__new__(cls)
    a._config = cfg
    a._agent_id = uuid4()
    a._org_id = None
    a._approval_handler = None
    return a


class TestAnthropicLocalPolicyEvaluator:
    def test_local_mode_builds_local_evaluator(self):
        """The bug: local mode had NO evaluator, so policies never ran."""
        ev = _adapter(AnthropicAdapter)._build_policy_evaluator()
        assert isinstance(ev, SDKLocalPolicyEvaluator), (
            "local mode must get the YAML evaluator; None means policies.yaml is ignored"
        )

    @pytest.mark.parametrize(
        "mode,expected", [("dry_run", True), ("enforce", False)]
    )
    def test_local_mode_threads_dry_run(self, mode, expected):
        """FRD-020 A5: without this the local path raises instead of recording."""
        ev = _adapter(AnthropicAdapter, enforcement_mode=mode)._build_policy_evaluator()
        assert ev._is_dry_run is expected

    def test_cloud_mode_still_builds_backend_evaluator(self):
        """Regression: the cloud path must be unchanged."""
        ev = _adapter(AnthropicAdapter, api_key=_CLOUD_KEY)._build_policy_evaluator()
        assert isinstance(ev, PolicyEvaluator)

    def test_policy_enforcement_disabled_returns_none(self):
        ev = _adapter(AnthropicAdapter, policies=False)._build_policy_evaluator()
        assert ev is None

    def test_policy_disabled_wins_over_local_mode(self):
        ev = _adapter(
            AnthropicAdapter, enforcement_mode="dry_run", policies=False
        )._build_policy_evaluator()
        assert ev is None


class TestAdapterParity:
    """wrap() dispatches to Anthropic/OpenAI through one branch — they must agree.

    The original defect was precisely a divergence here: OpenAI handled local
    mode, Anthropic did not.
    """

    @pytest.mark.parametrize("mode", ["dry_run", "enforce"])
    def test_both_adapters_build_local_evaluator(self, mode):
        an = _adapter(AnthropicAdapter, enforcement_mode=mode)._build_policy_evaluator()
        oa = _adapter(OpenAIAdapter, enforcement_mode=mode)._build_policy_evaluator()
        assert type(an) is type(oa) is SDKLocalPolicyEvaluator
        assert an._is_dry_run == oa._is_dry_run

    def test_both_adapters_build_cloud_evaluator(self):
        an = _adapter(AnthropicAdapter, api_key=_CLOUD_KEY)._build_policy_evaluator()
        oa = _adapter(OpenAIAdapter, api_key=_CLOUD_KEY)._build_policy_evaluator()
        assert type(an) is type(oa) is PolicyEvaluator

    def test_both_adapters_respect_disabled_policies(self):
        an = _adapter(AnthropicAdapter, policies=False)._build_policy_evaluator()
        oa = _adapter(OpenAIAdapter, policies=False)._build_policy_evaluator()
        assert an is oa is None
