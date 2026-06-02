# Security Policy

## Supported versions

Ardon-R2 is pre-1.0 and moves fast. Security fixes land on the latest
release line only.

| Version | Supported |
|---------|-----------|
| 0.2.x   | ✅ |
| < 0.2   | ❌ (please upgrade) |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately via GitHub's confidential advisory flow:

1. Go to the repository's **Security** tab →
   **"Report a vulnerability"** (GitHub Private Vulnerability Reporting).
2. If that is unavailable, send a direct message to the maintainer
   **@devendratandle** on GitHub asking for a private channel.

Please include:

- A description of the issue and its impact.
- A minimal R2 script or steps to reproduce.
- The R2 version (`r2 --version` or the GUI banner) and your OS.

## What to expect

As a single-maintainer project, response is best-effort but taken
seriously. You can expect an initial acknowledgement, an assessment of
severity, and — for confirmed issues — a fix on the current release
line and a credit in the release notes (unless you prefer to remain
anonymous).

## Scope notes

Ardon-R2 executes user-supplied R2 scripts by design; running an
untrusted script is equivalent to running untrusted code and is out of
scope. In-scope concerns include memory-safety defects reachable from
ordinary (non-malicious) input, crashes on well-formed scripts, and any
unexpected file/network access by the interpreter or GUI.
