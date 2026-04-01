# Tests for Clyro SDK Generic Adapter
# Implements PRD-001

"""Unit tests for the generic adapter."""



from clyro.adapters.generic import GenericAdapter, detect_adapter
from clyro.config import ClyroConfig
from clyro.session import Session
from clyro.trace import EventType, Framework


class TestGenericAdapter:
    """Tests for GenericAdapter class."""

    def test_init_with_function(self):
        """Test initializing adapter with a function."""
        def my_agent(x: int) -> int:
            return x * 2

        config = ClyroConfig()
        adapter = GenericAdapter(my_agent, config)

        assert adapter.agent == my_agent
        assert adapter.name == "my_agent"
        assert adapter.framework == Framework.GENERIC
        assert adapter.framework_version == "1.0.0"

    def test_init_with_lambda(self):
        """Test initializing adapter with a lambda."""
        agent = lambda x: x + 1  # noqa: E731
        config = ClyroConfig()
        adapter = GenericAdapter(agent, config)

        assert "<lambda>" in adapter.name

    def test_before_call_returns_context(self):
        """Test before_call hook returns context."""
        def my_agent() -> str:
            return "test"

        config = ClyroConfig()
        adapter = GenericAdapter(my_agent, config)
        session = Session(config=config)
        session.start()

        context = adapter.before_call(session, (), {})

        assert "start_time" in context
        assert "step_number" in context
        assert context["step_number"] == 1

    def test_after_call_creates_step_event(self):
        """Test after_call creates a proper step event."""
        def my_agent() -> str:
            return "result"

        config = ClyroConfig()
        adapter = GenericAdapter(my_agent, config)
        session = Session(config=config)
        session.start()

        context = {"start_time": 0.0, "step_number": 1}
        event = adapter.after_call(session, "result", context)

        assert event.event_type == EventType.STEP
        assert event.session_id == session.session_id
        assert event.step_number == 1
        assert "my_agent_complete" in event.event_name

    def test_after_call_captures_output_when_enabled(self):
        """Test that output is captured when enabled."""
        def my_agent() -> dict:
            return {"key": "value"}

        config = ClyroConfig(capture_outputs=True)
        adapter = GenericAdapter(my_agent, config)
        session = Session(config=config)
        session.start()

        context = {"start_time": 0.0, "step_number": 1}
        event = adapter.after_call(session, {"key": "value"}, context)

        assert event.output_data is not None
        assert event.output_data["result"]["key"] == "value"

    def test_after_call_no_output_when_disabled(self):
        """Test that output is not captured when disabled."""
        def my_agent() -> str:
            return "secret"

        config = ClyroConfig(capture_outputs=False)
        adapter = GenericAdapter(my_agent, config)
        session = Session(config=config)
        session.start()

        context = {"start_time": 0.0, "step_number": 1}
        event = adapter.after_call(session, "secret", context)

        assert event.output_data is None

    def test_on_error_creates_error_event(self):
        """Test on_error creates an error event."""
        def my_agent() -> None:
            raise ValueError("test error")

        config = ClyroConfig()
        adapter = GenericAdapter(my_agent, config)
        session = Session(config=config)
        session.start()

        error = ValueError("test error")
        context = {"start_time": 0.0, "step_number": 1}
        event = adapter.on_error(session, error, context)

        assert event.event_type == EventType.ERROR
        assert event.error_type == "ValueError"
        assert event.error_message == "test error"
        assert event.error_stack is not None

    def test_serialize_result_handles_primitives(self):
        """Test serialization of primitive types."""
        def my_agent() -> None:
            pass

        config = ClyroConfig()
        adapter = GenericAdapter(my_agent, config)

        assert adapter._serialize_result(None) == {"result": None}
        assert adapter._serialize_result("string") == {"result": "string"}
        assert adapter._serialize_result(42) == {"result": 42}
        assert adapter._serialize_result(3.14) == {"result": 3.14}
        assert adapter._serialize_result(True) == {"result": True}

    def test_serialize_result_handles_collections(self):
        """Test serialization of collections."""
        def my_agent() -> None:
            pass

        config = ClyroConfig()
        adapter = GenericAdapter(my_agent, config)

        assert adapter._serialize_result([1, 2, 3]) == {"result": [1, 2, 3]}
        assert adapter._serialize_result((1, 2)) == {"result": [1, 2]}
        assert adapter._serialize_result({"a": 1}) == {"result": {"a": 1}}

    def test_serialize_result_handles_pydantic_model(self):
        """Test serialization of Pydantic models."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            value: int

        def my_agent() -> None:
            pass

        config = ClyroConfig()
        adapter = GenericAdapter(my_agent, config)

        model = MyModel(value=42)
        result = adapter._serialize_result(model)

        assert result["result"]["value"] == 42

    def test_serialize_result_handles_custom_object(self):
        """Test serialization of custom objects."""
        class CustomObj:
            def __init__(self, x: int):
                self.x = x
                self._private = "hidden"

        def my_agent() -> None:
            pass

        config = ClyroConfig()
        adapter = GenericAdapter(my_agent, config)

        obj = CustomObj(10)
        result = adapter._serialize_result(obj)

        assert result["result"]["x"] == 10
        assert "_private" not in result["result"]

    def test_serialize_result_fallback_to_string(self):
        """Test serialization fallback to string."""
        class Unserializable:
            def __repr__(self) -> str:
                return "Unserializable()"

        # Remove __dict__ to test fallback
        def my_agent() -> None:
            pass

        config = ClyroConfig()
        adapter = GenericAdapter(my_agent, config)

        # For objects that can't be serialized, falls back to str()
        obj = object()  # Plain object has no __dict__ values
        result = adapter._serialize_result(obj)
        assert "result" in result


class TestDetectAdapter:
    """Tests for detect_adapter function."""

    def test_detect_generic_for_function(self):
        """Test detection returns generic for plain function."""
        def my_func() -> str:
            return "test"

        assert detect_adapter(my_func) == "generic"

    def test_detect_generic_for_callable_class(self):
        """Test detection returns generic for callable class."""
        class MyCallable:
            def __call__(self) -> str:
                return "test"

        assert detect_adapter(MyCallable()) == "generic"

    def test_detect_generic_for_lambda(self):
        """Test detection returns generic for lambda."""
        assert detect_adapter(lambda: None) == "generic"
