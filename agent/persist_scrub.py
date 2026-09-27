"""XP6F: scrub secrets and policy-blocked payloads BEFORE they are persisted to state.db.

Applied by hermes_state.SessionDB on every message INSERT (append_message and the bulk
replace/compact path). The FTS index tables are fed by triggers from messages.content,
so they only ever see the scrubbed text.

Layers (all run, in order):
  1. agent.redact.redact_sensitive_text(force=True)  - Hermes' own secret patterns
  2. strict header / bearer pass                       - sensitive headers and Bearer tokens -> [REDACTED]
     (Hermes' masker may keep a prefix/suffix; nothing of a credential is kept here)
  3. exact live-secret substitution                    - values of KEY/TOKEN/SECRET/PASSWORD variables in
                                                         ~/.hermes/.env (>=16 chars) -> [REDACTED]
Policy-blocked payloads: when a request is refused by a protected-project / DLP / egress policy,
its current-turn user/tool contents are registered (as SHA-256 only) and replaced by a denial
placeholder, both in rows already written for that session and on any later insert.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from typing import Any, Iterable, Optional

REDACTED = "[REDACTED]"
BLOCKED_PLACEHOLDER = "[NOT PERSISTED: content blocked by Elite policy]"
_HEADER_RE = re.compile(r"(?im)\b(authorization|proxy-authorization|x-api-key|api-key|cookie|set-cookie)(\"?\s*[:=]\s*\"?)([^\r\n\",}]+)")
_BEARER_RE = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=\-]{8,}")
_SECRET_NAME_RE = re.compile(r"(?i)(key|token|secret|password|passwd|credential)")
_env_cache: dict[str, Any] = {"mtime": None, "values": []}
_blocked: dict[str, set[str]] = {}
_lock = threading.Lock()


def _live_secret_values() -> list[str]:
    path = os.path.join(os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes")), ".env")
    try:
        m = os.path.getmtime(path)
    except OSError:
        return []
    if _env_cache["mtime"] != m:
        vals = []
        try:
            for line in open(path, encoding="utf-8", errors="ignore"):
                if "=" not in line or line.lstrip().startswith("#"):
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if len(v) >= 16 and _SECRET_NAME_RE.search(k) and not v.startswith(("/", "~", "http")):
                    vals.append(v)
        except OSError:
            pass
        _env_cache.update(mtime=m, values=sorted(set(vals), key=len, reverse=True))
    return _env_cache["values"]


def scrub_text(value: Any) -> Any:
    """Return *value* with secrets removed. Non-strings are returned unchanged."""
    if not isinstance(value, str) or not value:
        return value
    try:
        from agent.redact import redact_sensitive_text
        value = redact_sensitive_text(value, force=True)
    except Exception:
        pass   # layers 2 and 3 still run
    value = _HEADER_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", value)
    value = _BEARER_RE.sub(lambda m: f"{m.group(1)} {REDACTED}", value)
    for secret in _live_secret_values():
        if secret in value:
            value = value.replace(secret, REDACTED)
    return value


def _h(content: Any) -> Optional[str]:
    if not isinstance(content, str) or not content:
        return None
    return hashlib.sha256(content.encode("utf-8", "ignore")).hexdigest()


def mark_blocked(session_id: str, contents: Iterable[Any]) -> list[str]:
    """Register policy-blocked contents (SHA-256 only) for *session_id*; returns the raw strings for purging."""
    raw = [c for c in contents if isinstance(c, str) and c]
    with _lock:
        s = _blocked.setdefault(session_id or "", set())
        for c in raw:
            s.add(_h(c)); s.add(_h(scrub_text(c)))
    return raw


def is_blocked(session_id: str, content: Any) -> bool:
    h = _h(content)
    if h is None:
        return False
    with _lock:
        return h in _blocked.get(session_id or "", set())


def current_turn_contents(messages: list) -> list:
    """User/tool contents of the CURRENT turn (from the last user message on) - the part a policy refused."""
    last_user = max((i for i, m in enumerate(messages or []) if isinstance(m, dict) and m.get("role") == "user"), default=None)
    if last_user is None:
        return []
    return [m.get("content") for m in messages[last_user:] if isinstance(m, dict) and m.get("role") in ("user", "tool")]
