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

    config = ClyroConfig()
    rec_cfg = config.policy_recommender
    transport = args.llm_transport or rec_cfg.llm_transport
    deployment_mode = "cloud" if config.mode == "cloud" else "self-hosted"

    try:
        result = Recommender(base_url=rec_cfg.dashboard_base_url).suggest(
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
            f"Error: could not reach the catalogue at {rec_cfg.dashboard_base_url} "
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
        link = f"{rec_cfg.dashboard_base_url}/agents/new"
        print(
            f"\nOpen in wizard: {link} (use --json --out <file> to save the payload).",
            file=sys.stderr,
        )

    if args.apply:
        return _handle_apply()
    return 0


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
