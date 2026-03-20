# Clyro — OpenTelemetry Usage Guide

There are two independent OTEL paths. Use one or both depending on your setup.

---

## Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│ Path 1 — SDK Export                                                   │
│                                                                       │
│  clyro.wrap() ──→ Clyro backend (native)                             │
│              └──→ OTLPExporter ──→ any OTLP backend                  │
│                                   (Grafana Tempo, Jaeger, Datadog…)  │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ Path 2 — Direct OTEL Ingestion                                        │
│                                                                       │
│  OTEL-instrumented agent ──→ OTLPSpanExporter                        │
│                                    └──→ POST /v1/traces/otlp (HTTP)  │
│                                    └──→ gRPC :4317                   │
│                                         Clyro backend                 │
└───────────────────────────────────────────────────────────────────────┘
```

**Path 1** — You use the Clyro SDK (`clyro.wrap()`) and want traces mirrored to an external OTEL backend such as Grafana Tempo, Jaeger, or Honeycomb.

**Path 2** — Your agent is already instrumented with OpenTelemetry (e.g. via `opentelemetry-instrumentation-langchain`). You send traces directly to Clyro's OTLP endpoint without using `clyro.wrap()` at all.

---

## Path 1 — SDK OTLP Export

### What it does

When you wrap an agent with `clyro.wrap()`, traces normally go only to the Clyro backend. Adding `otlp_export_endpoint` causes every session's events to also be exported (in parallel, non-blocking) to any OTLP-compatible destination.

### Install

```bash
pip install clyro opentelemetry-proto
```

`opentelemetry-proto` provides the Protobuf definitions used for OTLP serialisation. It is imported lazily — the overhead is zero when export is disabled.

### One-liner integration

Add `otlp_export_endpoint` to your existing `ClyroConfig`. That's the only change.

**Before:**
```python
import os
import clyro

wrapped = clyro.wrap(
    my_agent,
    config=clyro.ClyroConfig(
        api_key=os.environ.get("CLYRO_API_KEY"),
        agent_name="my-agent",
    ),
)
```

**After:**
```python
import os
import clyro

wrapped = clyro.wrap(
    my_agent,
    config=clyro.ClyroConfig(
        api_key=os.environ.get("CLYRO_API_KEY"),
        agent_name="my-agent",
        otlp_export_endpoint="http://localhost:4318/v1/traces",  # added
    ),
)
```

The same pattern works with any framework:

```python
import os
# LangGraph
wrapped = clyro.wrap(compiled_graph, config=clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-graph",
    otlp_export_endpoint="https://tempo.mycompany.com/v1/traces",
))

# CrewAI
wrapped = clyro.wrap(crew, config=clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-crew",
    otlp_export_endpoint="https://tempo.mycompany.com/v1/traces",
))

# Generic function
@clyro.wrap(config=clyro.ClyroConfig(
    api_key=os.environ.get("CLYRO_API_KEY"),
    agent_name="my-agent",
    otlp_export_endpoint="https://tempo.mycompany.com/v1/traces",
))
def my_agent(query: str) -> str:
    ...
```

### Destination-specific examples

**Grafana Tempo (local):**
```python
otlp_export_endpoint="http://localhost:4318/v1/traces"
```

**Grafana Cloud:**
```python
otlp_export_endpoint="https://tempo-prod-xxx.grafana.net/tempo/otlp/v1/traces",
otlp_export_headers={
    "Authorization": "Basic <base64(instance_id:api_token)>",
},
```

**Honeycomb:**
```python
otlp_export_endpoint="https://api.honeycomb.io/v1/traces",
otlp_export_headers={
    "x-honeycomb-team": "<honeycomb-api-key>",
    "x-honeycomb-dataset": "my-agent",
},
```

**Jaeger (local):**
```python
otlp_export_endpoint="http://localhost:4318/v1/traces"
```

### OTLP export config reference

All fields are optional. Export is disabled when `otlp_export_endpoint` is not set.

| Field | Default | Description |
|---|---|---|
| `otlp_export_endpoint` | `None` | OTLP/HTTP endpoint. Set this to enable export. HTTPS required for remote hosts; HTTP allowed for localhost. |
| `otlp_export_headers` | `{}` | Extra HTTP headers (e.g. auth tokens for cloud backends). |
| `otlp_export_timeout_ms` | `5000` | Per-request timeout in milliseconds. |
| `otlp_export_queue_size` | `100` | In-memory queue capacity (batches). Drops oldest batch when full. |
| `otlp_export_compression` | `"gzip"` | Payload compression: `"gzip"` or `"none"`. |

Environment variable equivalents:

| Environment variable | Config field |
|---|---|
| `CLYRO_OTLP_EXPORT_ENDPOINT` | `otlp_export_endpoint` |
| `CLYRO_OTLP_EXPORT_TIMEOUT_MS` | `otlp_export_timeout_ms` |
| `CLYRO_OTLP_EXPORT_QUEUE_SIZE` | `otlp_export_queue_size` |
| `CLYRO_OTLP_EXPORT_COMPRESSION` | `otlp_export_compression` |

### Span mapping (Clyro → OTLP)

Each Clyro `TraceEvent` becomes one OTLP span:

| Clyro field | OTLP field |
|---|---|
| `session_id` | `trace_id` (UUID → 16 bytes) |
| `event_id` | `span_id` (lower 8 bytes of UUID) |
| `parent_event_id` | `parent_span_id` |
| `event_name` or `event_type` | `span.name` |
| `timestamp` | `start_time_unix_nano` |
| `timestamp + duration_ms` | `end_time_unix_nano` |
| `agent_stage` | `clyro.loop.stage` attribute (`think` / `act` / `observe`) |
| `agent_name` | `service.name` + `clyro.agent.name` resource attributes |
| `metadata.*` | span attributes (internal `_`-prefixed keys excluded) |
| `event_type == ERROR` | `status.code = STATUS_CODE_ERROR` |

### Failure behavior

Export is fully isolated from native Clyro ingest. If the OTLP backend is unreachable:
- The error is logged at `WARNING` level
- The native Clyro path is unaffected
- The agent call is unaffected
- No exception is propagated to the caller

---

## Path 2 — Direct OTEL Ingestion

### What it does

Send traces from any OTEL-instrumented code directly to Clyro's OTLP endpoint. No `clyro.wrap()` required. Clyro auto-registers the agent on first trace.

This is the right path if:
- You already use `opentelemetry-instrumentation-langchain`, `opentelemetry-instrumentation-crewai`, or any other OTEL auto-instrumentor
- You want to send traces from a non-Python agent or a service that already exports OTLP
- You prefer native OTEL tooling over the Clyro SDK

### Install

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

### Endpoints

| Transport | Endpoint | Auth header |
|---|---|---|
| HTTP (recommended) | `POST https://api.clyrohq.com/v1/traces/otlp` | `X-Clyro-API-Key: <key>` |
| gRPC | `api.clyrohq.com:4317` | `api-key: <key>` metadata |

Supported content types (HTTP): `application/x-protobuf` (default) and `application/json`.
Supported compression: `gzip` via `Content-Encoding: gzip`.

### One-liner integration

The only Clyro-specific requirement is setting `clyro.agent.name` in your OTEL resource. Everything else is standard OpenTelemetry.

Add this setup block to your agent file, **before** any framework imports:

```python
import os
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

resource = Resource.create({
    "service.name": "my-agent",
    "clyro.agent.name": "my-agent",   # required — Clyro uses this to identify your agent
})
provider = TracerProvider(resource=resource)
provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="https://api.clyrohq.com/v1/traces/otlp",
            headers={"X-Clyro-API-Key": os.environ.get("CLYRO_API_KEY")},
        )
    )
)
trace.set_tracer_provider(provider)
```

After this block, your agent runs unchanged.

### With LangChain / LangGraph auto-instrumentation

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
            opentelemetry-instrumentation-langchain
```

```python
import os
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Add this block before any LangChain imports
resource = Resource.create({
    "service.name": "my-langchain-agent",
    "clyro.agent.name": "my-langchain-agent",
})
provider = TracerProvider(resource=resource)
provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="https://api.clyrohq.com/v1/traces/otlp",
            headers={"X-Clyro-API-Key": os.environ.get("CLYRO_API_KEY")},
        )
    )
)
trace.set_tracer_provider(provider)
LangchainInstrumentor().instrument()  # auto-instruments all LLM calls, tool calls, chain runs

# --- Your existing agent code below, unchanged ---
from langchain_openai import ChatOpenAI
...
```

`LangchainInstrumentor` creates OTEL spans automatically for every LLM call, tool invocation, and chain step. No manual span code needed.

### With CrewAI auto-instrumentation

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
            opentelemetry-instrumentation-crewai
```

```python
import os
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Add this block before any CrewAI imports
resource = Resource.create({
    "service.name": "my-crew",
    "clyro.agent.name": "my-crew",
})
provider = TracerProvider(resource=resource)
provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="https://api.clyrohq.com/v1/traces/otlp",
            headers={"X-Clyro-API-Key": os.environ.get("CLYRO_API_KEY")},
        )
    )
)
trace.set_tracer_provider(provider)
CrewAIInstrumentor().instrument()

# --- Your existing crew code below, unchanged ---
from crewai import Agent, Task, Crew
...
```

### Manual spans (any Python code)

If you prefer to instrument manually rather than using an auto-instrumentor:

```python
tracer = trace.get_tracer("my-agent", "1.0.0")

with tracer.start_as_current_span("agent-session",
    attributes={
        "clyro.loop.stage": "think",   # optional — helps Clyro classify the span
        "user.query": "Summarise this document",
    }
) as session_span:

    with tracer.start_as_current_span("llm-call",
        attributes={"clyro.loop.stage": "think", "model": "gpt-4o"}
    ):
        response = llm.invoke(prompt)

    with tracer.start_as_current_span("tool-call",
        attributes={"clyro.loop.stage": "act", "tool": "vector_search"}
    ):
        results = retriever.get_relevant_documents(query)
```

### gRPC transport

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
```

```python
import os
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import grpc

exporter = OTLPSpanExporter(
    endpoint="api.clyrohq.com:4317",
    credentials=grpc.ssl_channel_credentials(),
    metadata=[("api-key", os.environ.get("CLYRO_API_KEY"))],
)
```

### The `clyro.loop.stage` attribute

Clyro uses this span attribute to classify spans into the Think / Act / Observe model visible in the UI.

| Value | Meaning |
|---|---|
| `think` | LLM reasoning, planning, or generation |
| `act` | Tool call, function execution, external API |
| `observe` | Processing results, embeddings, retrieval |

If this attribute is absent, Clyro falls back to mapping `gen_ai.operation.name` values:

| `gen_ai.operation.name` | Maps to |
|---|---|
| `chat`, `text_completion`, `create_agent` | `think` |
| `tool_call`, `execute_tool` | `act` |
| `embeddings`, `retrieve`, `rerank` | `observe` |

If neither attribute is set, the span is classified as `observe` by default.

### Span mapping (OTLP → Clyro)

| OTLP field | Clyro field |
|---|---|
| `trace_id` | `session_id` |
| `span_id` | `event_id` |
| `parent_span_id` | `parent_event_id` |
| `name` | `event_name` |
| `start_time_unix_nano` | `timestamp` |
| `end_time - start_time` | `duration_ms` |
| `status.code == ERROR` | `event_type = error` |
| `clyro.loop.stage` attribute | `agent_stage` |
| `clyro.agent.name` resource attr | agent lookup / auto-registration |
| all other attributes | stored in `metadata` |

### Agent auto-registration

On first trace, Clyro automatically creates an agent record using the `clyro.agent.name` resource attribute. The agent name is sanitised (lowercased, spaces → hyphens). Subsequent traces from the same name are linked to the same agent.

If you later register the same agent via the SDK (`clyro.wrap()` with the same `agent_name`), Clyro resolves them to the same agent record.

### Limits

| Limit | Value |
|---|---|
| Max spans per batch | 1,000 |
| Max payload size (decompressed) | 50 MB |
| Auth | `X-Clyro-API-Key` header (HTTP) or `api-key` metadata (gRPC) |

Batches exceeding 1,000 spans return HTTP 429 / gRPC `RESOURCE_EXHAUSTED`. Partial batches are accepted with a `partial_success` response — spans with missing `clyro.agent.name` are rejected individually while valid spans are accepted.

---

## Choosing between Path 1 and Path 2

| | Path 1 (SDK Export) | Path 2 (Direct Ingestion) |
|---|---|---|
| Uses `clyro.wrap()` | Yes | No |
| Clyro execution controls (step/cost limits) | Yes | No |
| Works with existing OTEL instrumentation | No | Yes |
| Sends to external OTEL backends | Yes | No |
| Required change | `otlp_export_endpoint` in `ClyroConfig` | OTEL setup block + `clyro.agent.name` resource attr |
| Extra install | `opentelemetry-proto` | `opentelemetry-sdk` + `opentelemetry-exporter-otlp-proto-http` |

Both paths store traces in the same Clyro backend (Redis → ClickHouse) and appear in the same Clyro UI.
