"""Unit tests for audit logging."""

import json

from clyro.hooks.audit import AuditLogger, redact_params


class TestRedactParams:
    def test_redacts_matching_keys(self):
        params = {"username": "admin", "password": "secret123"}
        result = redact_params(params, ["*password*"])
        assert result["username"] == "admin"
        assert result["password"] == "[REDACTED]"

    def test_redacts_nested_objects(self):
        params = {"auth": {"token": "abc123", "user": "bob"}}
        result = redact_params(params, ["*token*"])
        assert result["auth"]["token"] == "[REDACTED]"
        assert result["auth"]["user"] == "bob"

    def test_redacts_case_insensitive(self):
        params = {"API_KEY": "key123"}
        result = redact_params(params, ["*api_key*"])
        assert result["API_KEY"] == "[REDACTED]"

    def test_empty_params(self):
        assert redact_params(None, ["*password*"]) == {}
        assert redact_params({}, ["*password*"]) == {}

    def test_multiple_patterns(self):
        params = {"password": "p", "secret": "s", "name": "n"}
        result = redact_params(params, ["*password*", "*secret*"])
        assert result["password"] == "[REDACTED]"
        assert result["secret"] == "[REDACTED]"
        assert result["name"] == "n"

    def test_redacts_in_lists(self):
        params = {"items": [{"token": "abc"}, {"name": "bob"}]}
        result = redact_params(params, ["*token*"])
        assert result["items"][0]["token"] == "[REDACTED]"
        assert result["items"][1]["name"] == "bob"


class TestAuditLogger:
    def test_writes_jsonl_entry(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        al = AuditLogger(log_path=path)
        al.log_pre_tool_use(
            session_id="s1",
            tool_name="Bash",
            decision="allow",
            step_number=1,
            accumulated_cost_usd=0.001,
            tool_input={"command": "ls"},
        )
        al.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "pre_tool_use"
        assert entry["session_id"] == "s1"
        assert entry["decision"] == "allow"

    def test_writes_block_with_reason(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        al = AuditLogger(log_path=path)
        al.log_pre_tool_use(
            session_id="s1",
            tool_name="Bash",
            decision="block",
            step_number=5,
            accumulated_cost_usd=0.01,
            tool_input={"command": "rm -rf /"},
            reason="Policy violation: dangerous command",
        )
        al.close()

        entry = json.loads(path.read_text().strip())
        assert entry["decision"] == "block"
        assert "Policy violation" in entry["reason"]

    def test_redacts_sensitive_params(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        al = AuditLogger(log_path=path, redact_patterns=["*password*"])
        al.log_pre_tool_use(
            session_id="s1",
            tool_name="Bash",
            decision="allow",
            step_number=1,
            accumulated_cost_usd=0.0,
            tool_input={"command": "echo", "password": "secret123"},
        )
        al.close()

        entry = json.loads(path.read_text().strip())
        assert entry["tool_input"]["password"] == "[REDACTED]"
        assert entry["tool_input"]["command"] == "echo"

    def test_session_end_entry(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        al = AuditLogger(log_path=path)
        al.log_session_end(
            session_id="s1",
            total_steps=25,
            total_cost_usd=0.05,
            duration_seconds=120.5,
        )
        al.close()

        entry = json.loads(path.read_text().strip())
        assert entry["event"] == "session_end"
        assert entry["total_steps"] == 25

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "subdir" / "nested" / "audit.jsonl"
        al = AuditLogger(log_path=path)
        al.log_pre_tool_use(
            session_id="s1", tool_name="Bash", decision="allow",
            step_number=1, accumulated_cost_usd=0.0,
        )
        al.close()
        assert path.exists()

    def test_multiple_entries_append(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        al = AuditLogger(log_path=path)
        for i in range(3):
            al.log_pre_tool_use(
                session_id="s1", tool_name="Bash", decision="allow",
                step_number=i, accumulated_cost_usd=0.0,
            )
        al.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_pre_tool_use_includes_agent_id(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        al = AuditLogger(log_path=path)
        al.log_pre_tool_use(
            session_id="s1", tool_name="Bash", decision="allow",
            step_number=1, accumulated_cost_usd=0.0,
            agent_id="agent-123",
        )
        al.close()
        entry = json.loads(path.read_text().strip())
        assert entry["agent_id"] == "agent-123"

    def test_pre_tool_use_omits_agent_id_when_none(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        al = AuditLogger(log_path=path)
        al.log_pre_tool_use(
            session_id="s1", tool_name="Bash", decision="allow",
            step_number=1, accumulated_cost_usd=0.0,
        )
        al.close()
        entry = json.loads(path.read_text().strip())
        assert "agent_id" not in entry

    def test_post_tool_use_includes_agent_id(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        al = AuditLogger(log_path=path)
        al.log_post_tool_use(
            session_id="s1", tool_name="Bash",
            step_number=1, accumulated_cost_usd=0.0,
            agent_id="agent-456",
        )
        al.close()
        entry = json.loads(path.read_text().strip())
        assert entry["agent_id"] == "agent-456"

    def test_session_end_includes_agent_id(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        al = AuditLogger(log_path=path)
        al.log_session_end(
            session_id="s1", total_steps=10,
            total_cost_usd=0.05, duration_seconds=60.0,
            agent_id="agent-789",
        )
        al.close()
        entry = json.loads(path.read_text().strip())
        assert entry["agent_id"] == "agent-789"
