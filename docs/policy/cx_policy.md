# Clyro — CX Policy Usage Guide

Apply CX-specific policy templates to your AI agents and connect CX platforms (Zendesk, Intercom) for real-time and post-hoc policy enforcement.

---

## Prerequisites

1. **Clyro SDK installed** in your agent codebase (`pip install clyro`)
2. **API key configured** — obtain from the Clyro dashboard under Organization → API Keys
3. **Agent wrapped** — your CX agent is instrumented with `clyro.wrap()` or the appropriate adapter (LangGraph, CrewAI, Claude Agent SDK)

> **Authentication:** The REST API uses JWT Bearer tokens (`Authorization: Bearer <JWT_TOKEN>`). The Python SDK uses your Clyro API key (`cly_...`) directly. Both authenticate against the same organization.

---

## Part 1 — Policy Templates

### View Available CX Templates

List all CX templates via the policy templates API:

```bash
curl -H "Authorization: Bearer <JWT_TOKEN>" \
     https://api.clyro.dev/v1/policy-templates
```

Response includes CX templates (prefixed `cx_`) and bundle kits:

```json
{
  "templates": [
    {
      "id": "cx_refund_threshold",
      "name": "CX Refund Threshold",
      "description": "Block refund tool calls where amount exceeds threshold",
      "category": "business_logic"
    },
    ...
  ],
  "cx_bundles": [
    {
      "id": "cx_refund_agent",
      "name": "CX Refund Agent Starter Kit",
      "templates": ["cx_refund_threshold", "cx_refund_approval_gate", "cx_pii_response_guard"],
      "use_case": "Refund processing agents"
    }
  ]
}
```

### Available Templates

| Template ID | Failure Pattern | Check Type | Default Threshold |
|-------------|----------------|------------|-------------------|
| `cx_refund_threshold` | Unauthorized refund | `tool_call` | amount > $200 → block |
| `cx_refund_approval_gate` | Unauthorized refund | `tool_call` | amount > $100 → requires_approval |
| `cx_escalation_required` | Missing escalation | `tool_call` | billing_dispute, account_security, legal → block |
| `cx_pii_response_guard` | PII exposure | `llm_call` | SSN, social security, credit card number → block |
| `cx_discount_cap` | Unbounded discount | `tool_call` | percentage > 30% → block |
| `cx_grounding_required` | Policy hallucination | `llm_call` | Ungrounded phrases → block |
| `cx_kb_attribution` | Ungrounded KB response | `llm_call` | Missing source attribution → block |

---

### Apply a Template to an Agent

Create a policy from a template and assign it to your agent:

```bash
# Step 1: Create a policy from the template rules
curl -X POST \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  https://api.clyro.dev/v1/policies \
  -d '{
    "name": "refund_threshold_guard",
    "description": "Block refunds over $200",
    "action_type": "tool_call",
    "rules": {
      "version": "1.0",
      "default_action": "allow",
      "rules": [
        {
          "id": "rule-cx-refund-max",
          "name": "cx_refund_max_amount",
          "condition": {
            "field": "amount",
            "operator": "max_value",
            "value": 200
          },
          "action": "block",
          "message": "Refund of ${{amount}} exceeds maximum threshold of $200"
        }
      ]
    }
  }'

# Step 2: Assign the policy to your agent
curl -X POST \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  https://api.clyro.dev/v1/agents/<AGENT_ID>/policy-assignments \
  -d '{"policy_id": "<POLICY_ID>"}'
```

---

### Customize Threshold Values

After creating a policy from a template, update the threshold.

> **Note:** `PUT` replaces the entire `rules` object — you must include the complete rules definition, not just the changed field.

```bash
# Change refund threshold from $200 to $500
curl -X PUT \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  https://api.clyro.dev/v1/policies/<POLICY_ID> \
  -d '{
    "rules": {
      "version": "1.0",
      "default_action": "allow",
      "rules": [
        {
          "id": "rule-cx-refund-max",
          "name": "cx_refund_max_amount",
          "condition": {
            "field": "amount",
            "operator": "max_value",
            "value": 500
          },
          "action": "block",
          "message": "Refund of ${{amount}} exceeds maximum threshold of $500"
        }
      ]
    }
  }'
```

---

### Verify Enforcement

Trigger a test violation to confirm the policy is working:

```python
import os
import clyro

session = clyro.init(api_key=os.environ.get("CLYRO_API_KEY"), agent_id="<AGENT_ID>")

# This should be blocked by cx_refund_threshold (amount > 200)
result = session.check_policy("tool_call", {
    "tool_name": "process_refund",
    "amount": 250,
    "reason": "customer request",
})

print(result)
# PolicyDecision(decision='block', message='Refund of $250 exceeds maximum threshold of $200')
```

---

### Recommended Tool Call Schema for CX Actions

For CX templates to evaluate correctly, your agent's tool calls should include these fields:

**Refund Actions:**
```python
{
    "tool_name": "process_refund",
    "amount": 150.00,        # Required by cx_refund_threshold, cx_refund_approval_gate
    "currency": "USD",
    "reason": "defective product"
}
```

**Escalation Actions:**
```python
{
    "tool_name": "escalate_ticket",
    "issue_type": "billing_dispute",  # Required by cx_escalation_required
    "priority": "high"
}
```

**Discount Actions:**
```python
{
    "tool_name": "apply_discount",
    "percentage": 25,         # Required by cx_discount_cap
    "reason": "loyalty program"
}
```

---

## Part 2 — Webhook Integration

Connect Zendesk or Intercom to Clyro for post-hoc policy analysis of AI agent responses.

### Overview

Clyro's CX webhook integration receives events from CX platforms, transforms them into a normalized schema, stores them in ClickHouse, and evaluates them against your configured policies asynchronously.

**Endpoint:** `POST /v1/webhooks/cx`

**Flow:**
1. CX platform sends webhook → Clyro endpoint
2. Clyro verifies HMAC signature
3. Payload is transformed into normalized CXEvent
4. Event is stored in ClickHouse (visible in dashboards)
5. Event is evaluated against assigned policies (async)

### Webhook Prerequisites

1. **Webhook signing secret** — configured per platform in Organization → Settings
2. **CX policies assigned** — at least one CX policy template applied to your agent (see Part 1)

---

### Configure Signing Secrets

Each platform requires a signing secret for HMAC-SHA256 verification. Set these in your organization settings:

```bash
# The webhook_signing_secrets field stores per-platform secrets
# This is configured via the organization settings API or dashboard
{
  "zendesk": "your_zendesk_signing_secret",
  "intercom": "your_intercom_signing_secret"
}
```

---

### Zendesk Setup

#### Create Webhook in Zendesk Admin

1. Go to **Zendesk Admin Center** → **Apps and integrations** → **Webhooks**
2. Click **Create webhook**
3. Configure:
   - **Name:** Clyro Policy Monitor
   - **Endpoint URL:** `https://api.clyro.dev/v1/webhooks/cx`
   - **Request method:** POST
   - **Request format:** JSON
   - **Authentication:** None (Clyro uses its own API key header)

#### Add Custom Headers

| Header | Value |
|--------|-------|
| `X-Clyro-API-Key` | Your Clyro API key (`cly_...`) |
| `X-Platform` | `zendesk` |

The `X-Webhook-Signature` header is automatically added by Zendesk.

#### Create Trigger

1. Go to **Admin Center** → **Objects and rules** → **Triggers**
2. Create a trigger with:
   - **Condition:** Ticket is updated, via = AI agent
   - **Action:** Notify webhook → Clyro Policy Monitor

#### Supported Zendesk Events

| Event Type | Description |
|-----------|-------------|
| `messaging.agent_reply` | AI agent sends a message in messaging |
| `ticket.updated` | Ticket updated with AI-generated comment (filtered to `via.source.rel = "ai_agent"`) |

---

### Intercom Setup

#### Create Webhook in Intercom

1. Go to **Intercom Developer Hub** → **Webhooks**
2. Add a webhook subscription:
   - **URL:** `https://api.clyro.dev/v1/webhooks/cx`
   - **Topics:** `conversation.admin.replied`

#### Add Custom Headers

Intercom does not support custom headers on outgoing webhooks. You need a lightweight proxy (e.g., AWS API Gateway, Cloudflare Worker, or an internal middleware) to inject these headers before forwarding to Clyro:

| Header | Value |
|--------|-------|
| `X-Clyro-API-Key` | Your Clyro API key |
| `X-Platform` | `intercom` |

The `X-Hub-Signature` header (format: `sha256=<hex>`) is automatically added by Intercom.

#### AI Agent Filtering

Clyro automatically filters Intercom events to only process AI-generated responses:
- `author.type` must be `"bot"`
- `from_ai_agent` must be `true`

Human admin replies are silently skipped.

---

### Verify Webhook Integration

#### Send a Test Webhook

```bash
# Generate HMAC signature
SECRET="your_signing_secret"
BODY='{"event_type":"messaging.agent_reply","conversation":{"id":"12345"},"message":{"content":"Hello, how can I help?","created_at":"2026-03-10T00:00:00Z"}}'
SIGNATURE=$(echo -n "$BODY" | openssl dgst -sha256 -hmac "$SECRET" | awk '{print $2}')

# Send test webhook
curl -X POST https://api.clyro.dev/v1/webhooks/cx \
  -H "Content-Type: application/json" \
  -H "X-Clyro-API-Key: $CLYRO_API_KEY" \
  -H "X-Platform: zendesk" \
  -H "X-Webhook-Signature: $SIGNATURE" \
  -d "$BODY"
```

Expected response:
```json
{"status": "accepted", "events_processed": 1}
```

After successful ingestion, events appear in the Clyro dashboard with `source=cx_webhook`. Policy violations (if any) appear in the policy violations view.

---

## Error Responses

| Status | Error | Cause |
|--------|-------|-------|
| 401 | `InvalidAPIKey` | Missing or invalid `X-Clyro-API-Key` |
| 401 | `SigningSecretNotConfigured` | No signing secret configured for the platform |
| 401 | `InvalidSignature` | HMAC signature verification failed |
| 400 | `UnsupportedPlatform` | `X-Platform` not `zendesk` or `intercom` |
| 413 | `PayloadTooLarge` | Payload exceeds 1MB |
| 415 | `UnsupportedMediaType` | Content-Type is not `application/json` |
| 429 | `RateLimitExceeded` | Over 1,000 events/hour per organization |

---

## Rate Limits

- **1,000 events per hour** per organization
- Rate-limited responses include a `retry_after` field (seconds until the limit resets)

---

## Security

- All webhook payloads are verified using **HMAC-SHA256** signatures before processing
- Signature verification uses constant-time comparison to prevent timing attacks
- Signing secrets are stored per-organization, per-platform
- Raw body bytes are used for signature verification (before JSON parsing)

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| 401 `InvalidSignature` on every request | Signing secret mismatch between platform and Clyro org settings | Re-copy the secret from the platform's webhook config into Organization → Settings |
| 401 `SigningSecretNotConfigured` | No secret set for this platform in Clyro | Set the signing secret via the organization settings API or dashboard |
| Events accepted but no policy violations appear | No CX policies assigned to the agent | Assign at least one CX policy template — see Part 1 above |
| Intercom events return 400 `UnsupportedPlatform` | `X-Platform` header not being injected by proxy | Verify your webhook proxy adds `X-Platform: intercom` before forwarding |
| Events appear in dashboard but with `source=unknown` | Missing `X-Platform` header | Ensure the header is set to `zendesk` or `intercom` |

---

## Limitations

- **`contains` operator**: Uses substring matching, not regex. "SSN" matches "SSN" anywhere in the text, but won't match "S.S.N." or "Social Security Number" unless explicitly listed.
- **PII detection**: Best-effort via substring matching. Not a replacement for dedicated PII scanning services. Add additional substrings to the `value` list in the template for broader coverage.
- **`output` field dependency**: Templates using `llm_call` check type with the `output` field (`cx_pii_response_guard`, `cx_grounding_required`, `cx_kb_attribution`) require SDK version with FRD-013 support. Ensure your SDK is up to date.
