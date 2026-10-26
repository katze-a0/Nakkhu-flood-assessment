"""Credentials for account-gated sources (plan 002 / 008), read at run time only.

A value comes from the environment variable of the same name, else from a ``KEY=value`` file:
the path in ``NAKKHU_SECRETS``, else ``~/.config/nakkhu/secrets.env``. Values are never written
to the manifest, ``source.json``, run records or messages: every URL or error text that might
carry one goes through :func:`redact` first.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

SECRETS_ENV = "NAKKHU_SECRETS"
DEFAULT_SECRETS = Path.home() / ".config" / "nakkhu" / "secrets.env"
REDACTED = "<redacted>"
# Query parameters whose values are credentials or short-lived signatures.
_SECRET_QUERY = re.compile(
    r"(?i)([?&](?:api_key|apikey|token|access_token|signature|key-pair-id|policy|"
    r"x-amz-[a-z-]+)=)[^&#\s'\"]*")


class MissingCredential(RuntimeError):
    """A required credential is not set (the message names the variable, never a value)."""


def secrets_file() -> Path:
    return Path(os.environ.get(SECRETS_ENV) or DEFAULT_SECRETS)


def _from_file(name: str, path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip().removeprefix("export ").strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None


def secret(name: str) -> str:
    """The credential ``name`` (environment first, then the secrets file)."""
    value = os.environ.get(name) or _from_file(name, secrets_file())
    if not value:
        raise MissingCredential(
            f"{name} is not set (environment or {SECRETS_ENV} / {DEFAULT_SECRETS})")
    return value


def redact(text: object, *values: str | None) -> str:
    """``text`` with every given secret value and every credential-like query value masked."""
    out = str(text)
    for value in values:
        if value:
            out = out.replace(value, REDACTED)
    return _SECRET_QUERY.sub(lambda m: m.group(1) + REDACTED, out)
