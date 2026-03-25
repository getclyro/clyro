# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in the Clyro SDK, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email **security@clyro.dev** with:

- A description of the vulnerability
- Steps to reproduce or a proof-of-concept
- The affected version(s)
- Any potential impact assessment

We will acknowledge receipt within 48 hours and aim to provide an initial assessment within 5 business days.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |

## Security Best Practices

When using the Clyro SDK:

- **Never hardcode API keys** in source code. Use environment variables (`CLYRO_API_KEY`) or a secrets manager.
- **Keep the SDK updated** to benefit from the latest security patches.
- **Review policy configurations** before deploying to production.
- **Use HTTPS endpoints** for all non-localhost communication (enforced by the SDK).
