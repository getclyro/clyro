# Contributing to Clyro SDK

Thank you for your interest in contributing to Clyro! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) or pip for dependency management

### Development Setup

```bash
git clone https://github.com/getclyro/clyro.git
cd clyro
pip install -e ".[dev]"
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=clyro --cov-report=term-missing

# Specific test file
pytest tests/sdk/test_config.py -v
```

### Code Quality

```bash
# Linting + formatting
ruff check clyro/ --fix
ruff format clyro/
```

## Project Structure

```
clyro/
  adapters/       # Framework adapters (LangGraph, CrewAI, Anthropic, Claude Agent SDK)
  backend/        # Cloud backend communication (HTTP client, sync, circuit breaker)
  hooks/          # Claude Code hooks integration
  mcp/            # MCP governance wrapper
  storage/        # Local SQLite storage + migrations
  workers/        # Background sync workers
  config.py       # Configuration models
  wrapper.py      # Core wrap() function
  local_policy.py # Local YAML policy evaluator
  local_logger.py # Terminal logger for local mode
  cli.py          # CLI entry points (clyro-sdk, clyro-hook, clyro-mcp)
```

## Internal Requirement Codes

You'll see comments like `# Implements PRD-002` or `# Implements FRD-SOF-003` throughout the codebase. These reference internal product and technical design documents used during development:

| Prefix | Meaning |
|--------|---------|
| `PRD-xxx` | Product Requirements Document |
| `FRD-xxx` | Functional Requirements Document |
| `FRD-SOF-xxx` | SDK Open-Source Foundation requirements |
| `FRD-HK-xxx` | Claude Code Hooks requirements |
| `TDD-xxx` or `TDD §x` | Technical Design Document sections |
| `NFR-xxx` | Non-Functional Requirements |

These codes are **internal references only** and do not affect functionality. They help the team trace code back to design decisions. You can safely ignore them when contributing.

## How to Contribute

### Reporting Issues

- Use [GitHub Issues](https://github.com/getclyro/clyro/issues) for bugs and feature requests
- Include: Python version, OS, SDK version, adapter used, and a minimal reproduction

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Ensure code passes linting (`ruff check clyro/`)
6. Submit a pull request

### Coding Conventions

- **License header**: All `.py` files must start with:
  ```python
  # Copyright 2026 Clyro Inc.
  # SPDX-License-Identifier: Apache-2.0
  ```
- **Type hints**: Use type annotations for all function signatures
- **Tests**: Maintain >85% test coverage for new code
- **Fail-open**: SDK code must never crash the user's agent. Wrap risky operations in try/except and degrade gracefully
- **No secrets**: Never log API keys, tokens, or user data. Use the redaction module for sensitive parameters

### Adding a New Framework Adapter

See `clyro/adapters/generic.py` for the minimal adapter interface. Each adapter should:

1. Implement the callback/hook pattern for its framework
2. Capture LLM calls, tool calls, and agent state transitions
3. Integrate with the Prevention Stack (cost, steps, loops, policies)
4. Include unit tests in `tests/sdk/test_<adapter_name>.py`

## Code of Conduct

Be respectful and constructive. We're building tools to make AI agents safer for everyone.

## License

By contributing to Clyro SDK, you agree that your contributions will be licensed under the Apache License 2.0.
