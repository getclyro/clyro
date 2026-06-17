# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Policy Recommender — `clyro suggest` CLI
# Implements policy-recommender FRD-PR-FE-001..005, 015, 016 (CLI surface)

"""``clyro suggest <import-path>`` — introspect an agent and recommend policies.

Output modes: human-readable (default), ``--json`` (machine-readable), ``--out``
(write JSON to a file). ``--llm-transport`` overrides the configured transport.
The wizard deep-link / ``--apply`` are best-effort and degrade gracefully when no
credentials are configured.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from clyro.recommender.recommender import Recommender, SuggestResult
from clyro.recommender.transport import (
    EXIT_CONFIG_ERROR,
    EXIT_TRANSPORT_ERROR,
    EXIT_TRANSPORT_UNAVAILABLE,
    EXIT_UNEXPECTED,
    VALID_TRANSPORTS,
    RecommenderConfigError,
    TransportError,
    TransportUnavailable,
)

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def add_suggest_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``suggest`` subcommand (FRD-PR-FE-001/004/016)."""
    p = subparsers.add_parser(
        "suggest", help="Recommend policies for an existing agent (policy-recommender)"
    )
    p.add_argument("agent", help="Import path to the agent, e.g. mypkg.app:agent")
    p.add_argument(
        "--llm-transport",
        choices=VALID_TRANSPORTS,
        default=None,
        help="auto: claude-code → anthropic-api → rule-based, first available wins.",
    )
    p.add_argument("--json", action="store_true", help="Emit the JSON payload to stdout.")
    p.add_argument("--out", metavar="FILE", default=None, help="Write the JSON payload to FILE.")
    p.add_argument(
        "--apply", action="store_true", help="Apply the recommendation via the wizard endpoint."
    )
    p.add_argument(
        "-y", "--yes", action="store_true", help="Skip the --apply confirmation prompt (CI)."
    )
    p.add_argument("--no-cache", action="store_true", help="Bypass the fingerprint cache.")
    p.add_argument(
        "--prefill",
        action="store_true",
        help="POST the recommendation to the backend and print a one-time "
        "wizard deep-link (?prefill=<token>). Requires a configured api_key.",
    )
    p.add_argument(
        "--agent-name",
        metavar="NAME",
        default=None,
        help="Re-recommend an EXISTING agent: derive its agent_id as "
        "uuid5(org_id, name) and tag the prefill with it. Use the same name you "
        "govern the agent under (config.agent_name / clyro.wrap). Omit for a "
        "new agent — a plain --prefill carries no agent_id.",
    )
    p.add_argument(
        "--agent-id",
        metavar="UUID",
        default=None,
        help="Re-recommend an EXISTING agent by its exact agent_id (overrides "
        "--agent-name). Omit for a new agent.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Log what introspection extracted (tools, system prompt, topology, "
        "model) to stderr. Off by default — do not enable in production.",
    )


def _resolve_agent(path: str) -> Any:
    """Import an agent from ``module:attr`` or ``module.attr`` (FRD-PR-FE-001)."""
    module_path, _, attr = path.partition(":")
    if not attr:
        module_path, _, attr = path.rpartition(".")
    if not module_path or not attr:
        raise ImportError(f"Could not parse import path '{path}'")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _render_human(result: SuggestResult) -> str:
    p = result.payload
    transport_label = {
        "claude-code": "Using Claude Code",
        "anthropic-api": "Using Anthropic API",
        "rule-based": "Rule-based only — install Claude Code or set ANTHROPIC_API_KEY for AI-assisted recommendations.",
    }.get(result.transport, f"Using {result.transport}")

    lines = [
        _c(f"{transport_label} · cache: {result.cache} · catalogue {p.catalogue_version}", "2"),
        "",
        _c("Detected agent type:", "1") + f" {p.detected_agent_type}",
    ]
    if p.alternative_agent_types:
        lines.append(f"  or: {', '.join(p.alternative_agent_types)}")
    lines.append("")
    lines.append(_c("Recommended kits:", "1"))
    for k in p.recommended_kits:
        fit = " (best-fit)" if k.partial_match else ""
        lines.append(f"  • {k.id}{fit} [{k.confidence}] — {k.rationale}")
    if not p.recommended_kits:
        lines.append("  (none)")
    lines.append("")
    lines.append(_c("Inferred concerns:", "1"))
    for c in p.inferred_concerns:
        lines.append(f"  • {c.id} [{c.confidence}] — {c.rationale}")
    if not p.inferred_concerns:
        lines.append("  (none)")
    if p.sector_hint:
        lines.append("")
        lines.append(f"Sector hint: {p.sector_hint}")
    return "\n".join(lines)


def _enable_introspection_debug() -> None:
    """Surface the introspector's DEBUG logs on stderr (``--debug``).

    Attaches a dedicated stderr handler to the ``clyro.recommender`` logger and
    raises it to DEBUG. Off unless requested, so production runs stay silent.
    """
    import logging

    rec_logger = logging.getLogger("clyro.recommender")
    if not any(getattr(h, "_clyro_debug", False) for h in rec_logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[clyro:debug] %(name)s %(message)s"))
        handler._clyro_debug = True  # type: ignore[attr-defined]
        rec_logger.addHandler(handler)
    rec_logger.setLevel(logging.DEBUG)
    rec_logger.propagate = False  # avoid double-printing via root handlers


def handle_suggest(args: argparse.Namespace) -> int:
    """Run the ``suggest`` command. Returns the process exit code."""
    from clyro.config import ClyroConfig

    if getattr(args, "debug", False):
        _enable_introspection_debug()

    # Pre-flight: --out writable before any LLM call (FRD-PR-FE-004).
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        if not os.path.isdir(out_dir) or not os.access(out_dir, os.W_OK):
            print(f"OUTPUT_PATH_NOT_WRITABLE: {args.out}", file=sys.stderr)
            return EXIT_CONFIG_ERROR

    try:
        agent = _resolve_agent(args.agent)
    except Exception as exc:
        print(f"Could not import '{args.agent}': {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    # from_env() so CLYRO_API_KEY / CLYRO_ENDPOINT / CLYRO_MODE are honoured
    # (ClyroConfig() alone is a plain BaseModel and ignores the environment).
    config = ClyroConfig.from_env()
    rec_cfg = config.policy_recommender
    transport = args.llm_transport or rec_cfg.llm_transport
    deployment_mode = "cloud" if config.mode == "cloud" else "self-hosted"

    try:
        # Catalogue (/v1/agent-types, /concerns, /kits) lives on the API, not the
        # dashboard — use config.endpoint, the same host the prefill POST targets.
        result = Recommender(base_url=config.endpoint).suggest(
            agent,
            llm_transport=transport,
            api_key=config.api_key,
            deployment_mode=deployment_mode,
            use_cache=not args.no_cache,
        )
    except RecommenderConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except TransportUnavailable as exc:
        print(f"Error: transport unavailable ({exc.reason}). {exc.remediation}", file=sys.stderr)
        return EXIT_TRANSPORT_UNAVAILABLE
    except TransportError as exc:
        print(f"Error: transport failed ({exc.path}): {exc.cause}", file=sys.stderr)
        return EXIT_TRANSPORT_ERROR
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Catalogue fetch failed and no local snapshot is cached (first run, offline).
        print(
            f"Error: could not reach the catalogue at {config.endpoint} "
            f"({type(exc).__name__}). Connect once to cache the catalogue, then "
            "offline re-runs will work.",
            file=sys.stderr,
        )
        return EXIT_CONFIG_ERROR
    except Exception as exc:  # final safety net — never dump a raw traceback
        print(
            f"Error: recommendation failed unexpectedly ({type(exc).__name__}: {exc}). "
            "Try --llm-transport rule-based, or report this.",
            file=sys.stderr,
        )
        return EXIT_UNEXPECTED

    payload_dict = result.payload.to_dict()

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload_dict, fh, indent=2)

    if args.json:
        print(json.dumps(payload_dict))
    else:
        print(_render_human(result))
        if args.out:
            print(f"\nWrote recommendation to {args.out}", file=sys.stderr)

    # Wizard deep-link — a real ?prefill=<token> when --prefill (and an api_key)
    # allow; otherwise the plain link. Printed to stderr so --json stdout stays clean.
    want_prefill = getattr(args, "prefill", False)
    # agent_id is attached ONLY for the re-recommend flow (an existing agent the
    # caller identifies via --agent-id or --agent-name). A plain --prefill is the
    # new-agent flow: it carries no agent_id — the wizard creates the agent.
    link, prefilled = _wizard_link(
        payload_dict,
        config,
        rec_cfg,
        want_prefill=want_prefill,
        agent_name=getattr(args, "agent_name", None),
        agent_id=getattr(args, "agent_id", None),
    )
    if prefilled:
        print(f"\nPre-fill token created. Open in wizard:\n  {link}", file=sys.stderr)
    else:
        print(f"\nOpen in wizard: {link}", file=sys.stderr)
        if not want_prefill:
            print(
                "  (add --prefill to create a one-time token, or --json/--out to save the payload)",
                file=sys.stderr,
            )

    if args.apply:
        return _handle_apply()
    return 0


class _PrefillError(Exception):
    """A non-fatal reason the prefill POST could not be made (reported, not raised)."""


def _create_prefill_token(
    payload: dict[str, Any],
    config: Any,
    agent_name: str | None = None,
    agent_id: str | None = None,
) -> str:
    """POST ``payload`` to ``/agent-setup/prefill`` and return the token.

    Resolves ``org_id`` from the configured api_key (no extra round-trip). For the
    re-recommend flow it tags the payload with a target ``agent_id`` — taken
    directly from ``agent_id`` when given, else derived as ``uuid5(org_id,
    agent_name)``. A new-agent prefill passes neither and sends no ``agent_id``.
    Raises ``_PrefillError`` with a user-readable reason on any failure.
    """
    from clyro.wrapper import _extract_org_id_from_jwt_api_key, _generate_agent_id_from_name

    if not config.api_key:
        raise _PrefillError("no api_key configured (set CLYRO_API_KEY or config.api_key)")
    org_id = _extract_org_id_from_jwt_api_key(config.api_key)
    if org_id is None:
        raise _PrefillError("could not resolve org_id from the api_key (local key?)")

    # Re-recommend only: attach a target agent_id when the caller identifies an
    # existing agent. Explicit --agent-id wins; otherwise derive from --agent-name
    # (matching clyro.wrap() auto-registration). A plain prefill sends no agent_id
    # and the wizard creates a brand-new agent.
    if not payload.get("agent_id"):
        if agent_id:
            payload = {**payload, "agent_id": str(agent_id)}
        elif agent_name:
            payload = {**payload, "agent_id": str(_generate_agent_id_from_name(agent_name, org_id))}

    from clyro import __version__

    url = f"{config.endpoint.rstrip('/')}/v1/organizations/{org_id}/agent-setup/prefill"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Clyro-API-Key": config.api_key,
            # Identify the client. urllib's default "Python-urllib/x" UA is
            # banned by Cloudflare's default bot rules (403, error code 1010).
            "User-Agent": f"clyro-sdk/{__version__}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 (configured endpoint)
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")[:300]
        except Exception:
            detail = ""
        raise _PrefillError(f"server returned {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise _PrefillError(f"could not reach {config.endpoint} ({type(exc).__name__})") from exc

    token = body.get("prefill_token") if isinstance(body, dict) else None
    if not token:
        raise _PrefillError("response did not include a prefill_token")
    return str(token)


def _wizard_link(
    payload: dict[str, Any],
    config: Any,
    rec_cfg: Any,
    want_prefill: bool,
    agent_name: str | None = None,
    agent_id: str | None = None,
) -> tuple[str, bool]:
    """Return (wizard_url, prefilled). Adds ?prefill=<token> when requested + possible."""
    base = rec_cfg.dashboard_base_url.rstrip("/")
    if not want_prefill:
        return f"{base}/agents/new", False
    try:
        token = _create_prefill_token(payload, config, agent_name=agent_name, agent_id=agent_id)
        return f"{base}/agents/new?prefill={token}", True
    except _PrefillError as exc:
        print(f"(prefill skipped: {exc})", file=sys.stderr)
        return f"{base}/agents/new", False


def _handle_apply() -> int:
    """``--apply`` (FRD-PR-FE-003).

    **Known limitation (documented, not silent):** direct CLI apply needs the
    org-scoped ``/agent-setup/apply`` round-trip, which requires resolving the
    caller's ``org_id`` (the api_key alone doesn't carry it client-side). Until
    that org-resolution path lands, ``--apply`` does not POST; it tells the user
    to apply via the wizard. No confirmation prompt is shown (it would imply an
    action that doesn't happen). Tracked alongside the frontend wizard work (B-1).
    """
    print(
        "Note: `--apply` is not yet wired for direct CLI apply (needs the "
        "org-scoped credential flow that ships with the wizard, B-1). "
        "Open the wizard link above and apply via Step 5/6 — your selections "
        "carry over and you review before anything is created.",
        file=sys.stderr,
    )
    return 0
