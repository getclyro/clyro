# Tests for FRD-013 SDK Output Field Extension
# Verifies that record_llm_call includes "output" in post-completion policy check

"""
Unit tests for the output field in LLM call policy checks.

FRD-013 adds a post-completion policy check with an `output` field to
enable response-content policies (PII detection, grounding, attribution).
"""

from unittest.mock import patch

from clyro.config import ClyroConfig
from clyro.session import Session


class TestRecordLlmCallOutputField:
    """Tests for the output field passed to check_policy during record_llm_call."""

    def _make_session(self):
        """Create a started session for testing."""
        config = ClyroConfig(agent_name="test-agent")
        session = Session(config=config)
        session.start()
        return session

    def test_record_llm_call_includes_output_field(self):
        """Verify check_policy is called with output in params."""
        session = self._make_session()

        with patch.object(session, "check_policy") as mock_check:
            session.record_llm_call(
                model="gpt-4",
                input_data={"prompt": "Hello"},
                output_data={"content": "Hi there!"},
                input_tokens=10,
                output_tokens=5,
            )

            mock_check.assert_called_once()
            call_args = mock_check.call_args
            action_type = call_args[0][0]
            params = call_args[0][1]

            assert action_type == "llm_call"
            assert "output" in params

    def test_record_llm_call_output_string(self):
        """String output_data is used as-is for the output field."""
        session = self._make_session()

        with patch.object(session, "check_policy") as mock_check:
            session.record_llm_call(
                model="gpt-4",
                input_data={"prompt": "Hello"},
                output_data="This is a plain string response",
                input_tokens=10,
                output_tokens=5,
            )

            mock_check.assert_called_once()
            params = mock_check.call_args[0][1]
            assert params["output"] == "This is a plain string response"

    def test_record_llm_call_output_dict(self):
        """Dict output_data extracts the content key for the output field."""
        session = self._make_session()

        with patch.object(session, "check_policy") as mock_check:
            session.record_llm_call(
                model="gpt-4",
                input_data={"prompt": "Hello"},
                output_data={"content": "Extracted content value"},
                input_tokens=10,
                output_tokens=5,
            )

            mock_check.assert_called_once()
            params = mock_check.call_args[0][1]
            assert params["output"] == "Extracted content value"

    def test_record_llm_call_output_dict_no_content_key(self):
        """Dict output_data without content key falls back to str(output_data)."""
        session = self._make_session()
        output_data = {"response": "some value", "status": "ok"}

        with patch.object(session, "check_policy") as mock_check:
            session.record_llm_call(
                model="gpt-4",
                input_data={"prompt": "Hello"},
                output_data=output_data,
                input_tokens=10,
                output_tokens=5,
            )

            mock_check.assert_called_once()
            params = mock_check.call_args[0][1]
            # When content key is missing, get("content", "") returns ""
            # which is falsy, so str(output_data) is used instead
            assert params["output"] == str(output_data)

    def test_record_llm_call_output_dict_empty_content(self):
        """Dict output_data with empty content key falls back to str(output_data)."""
        session = self._make_session()
        output_data = {"content": "", "metadata": "extra"}

        with patch.object(session, "check_policy") as mock_check:
            session.record_llm_call(
                model="gpt-4",
                input_data={"prompt": "Hello"},
                output_data=output_data,
                input_tokens=10,
                output_tokens=5,
            )

            mock_check.assert_called_once()
            params = mock_check.call_args[0][1]
            # Empty string content is falsy, so str(output_data) is used
            assert params["output"] == str(output_data)

    def test_record_llm_call_output_none(self):
        """None output_data uses empty string for the output field."""
        session = self._make_session()

        with patch.object(session, "check_policy") as mock_check:
            session.record_llm_call(
                model="gpt-4",
                input_data={"prompt": "Hello"},
                output_data=None,
                input_tokens=10,
                output_tokens=5,
            )

            mock_check.assert_called_once()
            params = mock_check.call_args[0][1]
            assert params["output"] == ""

    def test_output_field_in_policy_params(self):
        """Verify 'output' key exists in policy parameters alongside other fields."""
        session = self._make_session()

        with patch.object(session, "check_policy") as mock_check:
            session.record_llm_call(
                model="claude-3-sonnet",
                input_data={"messages": [{"role": "user", "content": "test"}]},
                output_data={"content": "response text"},
                input_tokens=50,
                output_tokens=25,
            )

            mock_check.assert_called_once()
            params = mock_check.call_args[0][1]

            # All expected policy parameter keys must be present
            assert "model" in params
            assert "cost" in params
            assert "step_number" in params
            assert "input" in params
            assert "output" in params

            # Verify correct values for each field
            assert params["model"] == "claude-3-sonnet"
            assert params["step_number"] == 1
            assert isinstance(params["cost"], float)

    def test_output_field_contains_response_text(self):
        """Verify the output contains the actual LLM response text."""
        session = self._make_session()
        expected_response = "The capital of France is Paris."

        with patch.object(session, "check_policy") as mock_check:
            session.record_llm_call(
                model="gpt-4",
                input_data={"prompt": "What is the capital of France?"},
                output_data={"content": expected_response},
                input_tokens=20,
                output_tokens=10,
            )

            mock_check.assert_called_once()
            params = mock_check.call_args[0][1]
            assert params["output"] == expected_response


class TestOutputTextExtraction:
    """Tests for the output text extraction logic used in record_llm_call.

    This class tests the extraction logic in isolation to verify all branches:
        if output_data is not None:
            if isinstance(output_data, dict):
                output_text = output_data.get("content", "") or str(output_data)
            elif isinstance(output_data, str):
                output_text = output_data
            else:
                output_text = str(output_data)
    """

    def _extract_output_text(self, output_data):
        """Replicate the output text extraction logic from session.py."""
        output_text = ""
        if output_data is not None:
            if isinstance(output_data, dict):
                output_text = output_data.get("content", "") or str(output_data)
            elif isinstance(output_data, str):
                output_text = output_data
            else:
                output_text = str(output_data)
        return output_text

    def test_none_returns_empty_string(self):
        assert self._extract_output_text(None) == ""

    def test_string_returned_as_is(self):
        assert self._extract_output_text("hello world") == "hello world"

    def test_empty_string_returned_as_is(self):
        assert self._extract_output_text("") == ""

    def test_dict_with_content_key(self):
        assert self._extract_output_text({"content": "abc"}) == "abc"

    def test_dict_without_content_key(self):
        data = {"response": "value"}
        assert self._extract_output_text(data) == str(data)

    def test_dict_with_empty_content_key(self):
        data = {"content": "", "other": "x"}
        # Empty string is falsy, so falls back to str(data)
        assert self._extract_output_text(data) == str(data)

    def test_dict_with_none_content_key(self):
        data = {"content": None, "other": "x"}
        # None is falsy, so falls back to str(data)
        assert self._extract_output_text(data) == str(data)

    def test_non_string_non_dict_uses_str(self):
        assert self._extract_output_text(42) == "42"
        assert self._extract_output_text(["a", "b"]) == "['a', 'b']"
        assert self._extract_output_text(True) == "True"

    def test_multiline_string(self):
        text = "line one\nline two\nline three"
        assert self._extract_output_text(text) == text

    def test_dict_with_long_content(self):
        long_content = "x" * 10000
        assert self._extract_output_text({"content": long_content}) == long_content
