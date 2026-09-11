"""External-safe prompt/context projection for outbound external-AI requests.

Hermes injects framework-owned material into every system prompt — the skills
index, memory, the user profile, context files. Some of that material references
projects that the owner's protected-project policy forbids sending to an
external provider. The Elite external gateway's DLP correctly refuses those
requests, which leaves Hermes unable to use an external model at all.

This module is *defense in depth*, not a replacement for gateway DLP. It removes
**unintended framework contamination** from the outbound request so a legitimate
request can proceed, and it FAILS CLOSED for everything else.

Design rules (deliberate, and load-bearing):

* **Structured, never substring-redaction.** Filtering happens at whole context
  blocks and at the entry level inside blocks Hermes itself delimits
  (``<available_skills>`` entries, ``§``-separated memory entries). A protected
  identifier is never snipped out of the middle of a sentence, a path or a code
  block, because that produces malformed instructions.
* **User content is never sanitized.** If the user's own message carries
  protected material, the request stays blocked. Sanitizing it would turn this
  module into a DLP bypass, which is the exact opposite of its purpose.
* **Tool results are never sanitized** either, with one narrow exception: a
  result produced by a *framework skill-reading tool* is framework content by
  definition, so it is replaced wholesale with a neutral marker. Any other tool
  result that matches fails closed.
* **Fail closed on any doubt** — unreadable policy, failed integrity baseline,
  too few rules, a parse error, or residue the structured pass could not
  attribute to a filterable block.
* **The local path is never touched.** Projection runs only for external targets.
* **No content is ever logged.** Records carry component types and counts.

Policy comes from the existing root-owned guard so there is exactly one
protected-policy interpretation on this machine; this module never maintains a
second copy of the patterns.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

#: Canonical, root-owned policy engine. Deliberately NOT configurable from the
#: environment: this module trusts the loaded guard for _POLICY_OK, _BASELINE_OK
#: and match(), so an env-substitutable path would let any process replace the
#: policy engine with a permissive stub and have policy_state() still report
#: healthy. Tests override the module attribute in-process instead, which
#: requires code execution inside Hermes rather than a writable env var.
GUARD_PATH = "/Library/Application Support/EliteProtected/elite_protected_guard.py"

#: Set False ONLY by tests, in-process. Production requires the guard to be
#: owned by root, so an unprivileged process cannot substitute it.
REQUIRE_ROOT_GUARD = True

#: Tool results produced by these tools are Hermes reading its OWN skill
#: library, i.e. framework content. They may be replaced with a neutral marker.
#: Anything not named here is treated as workspace data and fails closed.
FRAMEWORK_SKILL_TOOLS = frozenset({"skill_view", "skills_list", "skill_search"})

SKILL_BODY_MARKER = "Skill unavailable for external execution due to project policy."

#: The exact heading Hermes emits immediately before its own skills index
#: (agent/prompt_builder.py). Its presence is what distinguishes a
#: framework-emitted block from a look-alike in someone's context file.
_FRAMEWORK_SKILLS_PREAMBLE = "## Skills (mandatory)"

_SKILLS_BLOCK_RE = re.compile(r"(<available_skills>\n)(.*?)(\n?</available_skills>)", re.DOTALL)
_MEM_SEPARATOR = "═" * 46
#: A memory block runs from its header to the next separator block, or to a
#: blank line — whichever comes first. It must NOT be allowed to run to the end
#: of the prompt: the volatile tier appends the external-memory block and the
#: timestamp/model lines after the last memory block, and a block bounded by
#: ``\Z`` would treat all of that as one giant "entry" and delete it on a match.
_MEM_BLOCK_RE = re.compile(
    r"(" + _MEM_SEPARATOR + r"\n(?:MEMORY|USER PROFILE)[^\n]*\n" + _MEM_SEPARATOR + r"\n)"
    r"(.*?)"
    r"(?=\n\n|\n" + _MEM_SEPARATOR + r"\n|\Z)",
    re.DOTALL,
)
_MEM_ENTRY_DELIM = "\n§\n"

_lock = threading.Lock()
_guard_cache: Dict[str, Any] = {}

#: Low-cardinality in-process counters. Labels are bounded by design:
#: component types and decision/reason enums only — never a path, project,
#: filename, skill name, user or any prompt text.
COUNTERS: Dict[Tuple[str, ...], int] = {}


def _count(metric: str, **labels: str) -> None:
    key = (metric,) + tuple(f"{k}={v}" for k, v in sorted(labels.items()))
    with _lock:
        COUNTERS[key] = COUNTERS.get(key, 0) + 1


def metrics_text() -> List[str]:
    """Render counters in Prometheus text format (bounded labels only)."""
    out: List[str] = []
    with _lock:
        snapshot = sorted(COUNTERS.items())
    for key, val in snapshot:
        metric, labels = key[0], key[1:]
        lbl = ("{" + ",".join(f'{p.split("=", 1)[0]}="{p.split("=", 1)[1]}"'
                              for p in labels) + "}") if labels else ""
        out.append(f"{metric}{lbl} {val}")
    return out


# ---------------------------------------------------------------- policy


class PolicyUnavailable(Exception):
    """Raised when the protected policy cannot be trusted. Always fail closed."""


class ExternalProjectionBlocked(Exception):
    """Raised instead of sending a request that could not be made external-safe.

    Carries ``status_code = 400`` so Hermes' existing classifier treats it as a
    non-retryable client error (``FailoverReason.format_error``). 403 must NOT
    be used: the classifier maps 401/403 to ``FailoverReason.auth``, which sends
    the request into ``_recover_with_credential_pool`` — attempting credential
    refreshes and then marking healthy credentials exhausted and rotating
    through the user's whole pool, while advising them their API key was
    rejected. A refusal decided locally must never touch credentials.
    """

    status_code = 400

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(
            "blocked by Elite protected-project policy before leaving this machine "
            f"(reason: {reason}). The request was NOT sent to any external provider. "
            "Run this task on a local model, or remove the protected reference."
        )


def _deny_path() -> str:
    """Resolve the policy file the guard will read, the same way the guard does."""
    return os.environ.get(
        "ELITE_PROTECTED_DENY",
        os.path.join(os.path.dirname(GUARD_PATH), "protected_deny.yaml"),
    )


def _load_guard() -> Any:
    """Import the root-owned guard module by path, once per policy revision.

    The guard reads its patterns from protected_deny.yaml at IMPORT time, so the
    cache must key on that file as well as on the guard module itself. Keying on
    the module alone left a long-running process using a stale pattern set after
    the owner tightened the policy — policy changes are one-way tightening, so a
    stale cache is a genuine gap, not merely a refresh delay.

    A *tampered* policy is still rejected: the guard's own baseline check runs on
    every re-import and ``policy_state()`` refuses to trust the result.
    """
    try:
        gst = os.stat(GUARD_PATH)
    except OSError as exc:
        raise PolicyUnavailable("guard_unreadable") from exc
    if REQUIRE_ROOT_GUARD and gst.st_uid != 0:
        # A guard this process could have written is not a trust anchor.
        raise PolicyUnavailable("guard_not_root_owned")
    if REQUIRE_ROOT_GUARD:
        expected = _baseline_guard_sha()
        if expected:
            import hashlib as _hl
            try:
                actual = _hl.sha256(open(GUARD_PATH, "rb").read()).hexdigest()
            except OSError as exc:
                raise PolicyUnavailable("guard_unreadable") from exc
            if actual != expected:
                raise PolicyUnavailable("guard_hash_mismatch")
    try:
        dst = os.stat(_deny_path())
        deny_stamp: Tuple[int, int] = (dst.st_mtime_ns, dst.st_size)
    except OSError:
        # A missing policy file is not fatal here: the guard reports it through
        # _POLICY_OK and policy_state() then fails closed.
        deny_stamp = (0, 0)
    try:
        bst = os.stat(os.path.join(os.path.dirname(GUARD_PATH), "policy.baseline.json"))
        base_stamp: Tuple[int, int] = (bst.st_mtime_ns, bst.st_size)
    except OSError:
        base_stamp = (0, 0)
    stamp = (gst.st_mtime_ns, gst.st_size, deny_stamp, base_stamp)

    with _lock:
        if _guard_cache.get("stamp") == stamp:
            return _guard_cache["module"]

    spec = importlib.util.spec_from_file_location("_elite_protected_guard", GUARD_PATH)
    if spec is None or spec.loader is None:
        raise PolicyUnavailable("guard_unloadable")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise PolicyUnavailable("guard_import_failed") from exc

    with _lock:
        _guard_cache["stamp"] = stamp
        _guard_cache["module"] = mod
    return mod


def _baseline_guard_sha() -> Optional[str]:
    """The guard's expected SHA-256 from policy.baseline.json, if recorded.

    The baseline file already carries this for the out-of-band integrity
    monitor; checking it here as well puts guard substitution on the request
    path rather than relying on a periodic sweep.
    """
    try:
        with open(os.path.join(os.path.dirname(GUARD_PATH),
                               "policy.baseline.json")) as f:
            return json.load(f).get("guard_sha256") or None
    except Exception:
        return None


def policy_state() -> Dict[str, Any]:
    """Return the trust state of the protected policy.

    ``ok`` is True only when the guard loaded the on-disk rules AND its
    tamper-evident baseline verified. Either failing means the on-disk policy is
    not trustworthy, so external projection must refuse rather than rely on the
    guard's built-in fallback patterns.
    """
    try:
        g = _load_guard()
    except PolicyUnavailable as exc:
        return {"ok": False, "reason": str(exc), "pattern_count": 0}

    policy_ok = bool(getattr(g, "_POLICY_OK", False))
    baseline_ok = bool(getattr(g, "_BASELINE_OK", False))
    baseline_reason = str(getattr(g, "_BASELINE_REASON", "unknown"))
    patterns = getattr(g, "_PATTERNS", []) or []
    min_rules = int(getattr(g, "MIN_RULES", 20))

    if not policy_ok:
        return {"ok": False, "reason": "policy_not_ok", "pattern_count": len(patterns)}
    if not baseline_ok:
        return {"ok": False, "reason": f"baseline_{baseline_reason}",
                "pattern_count": len(patterns)}
    if len(patterns) < min_rules:
        return {"ok": False, "reason": "below_min_rules", "pattern_count": len(patterns)}
    return {"ok": True, "reason": "ok", "pattern_count": len(patterns)}


def _matcher():
    """Return the guard's ``match`` callable, or raise PolicyUnavailable."""
    state = policy_state()
    if not state["ok"]:
        raise PolicyUnavailable(state["reason"])
    return _load_guard().match


# ---------------------------------------------------------------- targets


#: Loopback inference servers Hermes itself ships defaults for. These are named
#: SERVICES on the loopback interface, not "a private address" — the distinction
#: matters, because inferring locality from address shape is what let an
#: arbitrary RFC1918 or tailnet host skip projection.
_BUILTIN_LOOPBACK_ROUTES = tuple(
    (h, p) for h in ("127.0.0.1", "localhost", "::1")
    for p in ("11434", "1234")
)

#: Hosts that are external providers no matter where they appear.
_PUBLIC_PROVIDER_HOSTS = (
    "api.openai.com", "openai.azure.com", "api.anthropic.com", "api.deepseek.com",
    "openrouter.ai", "generativelanguage.googleapis.com", "api.z.ai",
    "open.bigmodel.cn", "api.x.ai", "api.groq.com", "api.mistral.ai",
    "api.kimi.com", "integrate.api.nvidia.com", "inference-api.nousresearch.com",
    "api.together.xyz", "api.fireworks.ai", "bedrock-runtime", "chatgpt.com",
    "api.stepfun.com", "api.arcee.ai", "githubcopilot.com",
)

#: The Elite external AI gateway. External BY DEFINITION even on loopback: it is
#: the sanctioned egress, so a request to it must be projected.
_ELITE_GATEWAY_PORT = "8082"


def _host_port(url: Any) -> Tuple[str, str]:
    """Split a URL into (host, port) using the same semantics as the HTTP client.

    Parsed with urlsplit rather than a hand-rolled split so userinfo cannot
    smuggle an approved host: ``http://127.0.0.1:11434@evil.example.com`` must
    read as ``evil.example.com``, which is what httpx will actually dial.
    Returns ("", "") when the URL is unparseable or its authority is ambiguous,
    and the caller treats that as EXTERNAL.
    """
    if not isinstance(url, str):
        return "", ""
    u = url.strip()
    if not u:
        return "", ""
    probe = re.sub(r"^[A-Za-z0-9+.\-]+://", "", u, count=1)
    authority = probe.split("/", 1)[0]
    if any(c in authority for c in ("@", "\\", "#", "?", " ", "\t", "\n", "\r", "\x00")):
        return "", ""
    try:
        from urllib.parse import urlsplit
        parts = urlsplit(u if "://" in u else "//" + u)
        host = (parts.hostname or "").lower()
        port = str(parts.port) if parts.port is not None else ""
    except ValueError:
        return "", ""
    return host, port


#: Hostnames accepted as local. Loopback service names only — an arbitrary DNS
#: name in config is NOT evidence of locality: it resolves through the search
#: domain and can point anywhere.
_LOCAL_HOST_NAMES = ("localhost",)


def _is_local_literal(host: str) -> bool:
    """True only for a host that is positively local BY ADDRESS, not by name.

    Being named in the configuration is permission to skip projection, never
    evidence that the destination is local. Without this check a single edited
    config line (``https://evil.example.com:4000``) joined the approved-route set
    and switched projection off for that destination.

    Denying by a list of known public hosts would be the wrong direction — a new
    or unlisted provider would stay approved by default — so this allows only
    address classes that cannot be a public provider.
    """
    if not host:
        return False
    h = host.strip("[]").lower()
    if h in _LOCAL_HOST_NAMES:
        return True
    try:
        import ipaddress
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False            # a DNS name is never evidence of locality
    # Loopback is decided FIRST: ``::1`` sits inside ``::/8``, which ipaddress
    # reports as reserved, so testing is_reserved before it would reject the
    # most obviously local address there is.
    if ip.is_loopback:
        return True
    # Deliberately NOT local, mirroring the auxiliary-egress model:
    #   link-local  — 169.254.169.254 is the cloud metadata service, and
    #                 treating it as an approved route exposes instance creds;
    #   6to4/teredo — report as "private" to ipaddress but EMBED an arbitrary
    #                 IPv4, including public addresses and that same metadata IP;
    #   unspecified — 0.0.0.0 / :: is a bind-all address, not a destination.
    if ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
        return False
    if ip.version == 6:
        for embedded in ("2002::/16", "2001::/23"):
            if ip in ipaddress.ip_network(embedded):
                return False
    if ip.is_private:
        return True
    # CGNAT / shared address space (100.64.0.0/10) is where Tailscale hands out
    # addresses and how the Elite cluster is reached. ipaddress does NOT report
    # it as private, so name it explicitly.
    return ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10")


def _configured_local_routes() -> List[Tuple[str, str]]:
    """(host, port) routes for local inference NAMED IN HERMES' CONFIGURATION.

    A route is trusted only when BOTH hold:

    * it is declared — in ``providers:``, ``custom_providers:`` or
      ``model.base_url``. Address shape alone is not enough: treating any
      RFC1918/CGNAT target as local let an arbitrary private endpoint, including
      any peer on the tailnet, skip projection.
    * it names a local address literal (``_is_local_literal``). Configuration
      alone is not enough either: it is writable, so a configured
      ``https://evil.example.com:4000`` would otherwise be approved outright.

    Reads Hermes' own canonical config accessors only, so JOB C stands alone.
    """
    routes: List[Tuple[str, str]] = list(_BUILTIN_LOOPBACK_ROUTES)

    def _add(raw: Any) -> None:
        host, port = _host_port(str(raw or ""))
        if not host or not port:
            return                      # a portless route would approve a whole host
        if not _is_local_literal(host):
            return                      # configured, but not positively local
        if any(pub in host for pub in _PUBLIC_PROVIDER_HOSTS):
            return                      # belt and braces; _is_local_literal wins
        if port == _ELITE_GATEWAY_PORT:
            return                      # the sanctioned gateway is external
        routes.append((host, port))

    try:
        from hermes_cli.config import load_config_readonly as _load
    except Exception:
        try:
            from hermes_cli.config import load_config as _load
        except Exception:
            return routes
    try:
        cfg = _load() or {}
    except Exception:
        return routes
    if not isinstance(cfg, dict):
        return routes

    for spec in (cfg.get("providers") or {}).values():
        if isinstance(spec, dict):
            _add(spec.get("base_url") or spec.get("api_base"))
    legacy = cfg.get("custom_providers")
    if isinstance(legacy, list):
        for spec in legacy:
            if isinstance(spec, dict):
                _add(spec.get("base_url") or spec.get("api_base"))
    elif isinstance(legacy, dict):
        for spec in legacy.values():
            if isinstance(spec, dict):
                _add(spec.get("base_url") or spec.get("api_base"))
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        prov = str(model_cfg.get("provider") or "")
        if prov == "custom" or prov.startswith("custom:"):
            _add(model_cfg.get("base_url"))
    return routes


def is_external_target(provider: Optional[str], base_url: Optional[str]) -> bool:
    """Is this request leaving for a target that must be projected?

    True (project it) unless the destination matches a CONFIGURED local
    inference route. Deterministic: no network call, no model call.

    Everything unrecognised is external. That is the safe direction — the worst
    case is that a local request has framework blocks removed, never that a
    request to an unvetted endpoint escapes projection.
    """
    # Coerce explicitly: these come off the agent object, which is a Mock in
    # parts of the test suite, and a non-str would otherwise raise inside re.
    url = str(base_url or "").strip()
    prov = str(provider or "").strip().lower()

    host, port = _host_port(url)
    if not host and prov:
        # No URL given: resolve the provider name through configuration rather
        # than trusting the name itself.
        try:
            from hermes_cli.config import load_config_readonly as _load
            spec = ((_load() or {}).get("providers") or {}).get(prov)
            if isinstance(spec, dict):
                host, port = _host_port(
                    str(spec.get("base_url") or spec.get("api_base") or ""))
        except Exception:
            host, port = "", ""
    if not host:
        return True                     # unknown / ambiguous -> project

    if port == _ELITE_GATEWAY_PORT:
        return True                     # sanctioned gateway is external egress
    if any(pub in host for pub in _PUBLIC_PROVIDER_HOSTS):
        return True

    for ahost, aport in _configured_local_routes():
        if host == ahost and port == aport:
            return False                # configured local inference route
    return True


# ---------------------------------------------------------------- filtering


class Projection:
    """Result of projecting one outbound request."""

    def __init__(self) -> None:
        self.allowed: bool = True
        self.reason: str = "ok"
        self.messages: Optional[List[Dict[str, Any]]] = None
        self.excluded: Dict[str, int] = {}
        self.policy_reason: str = "ok"

    def _exclude(self, component: str, n: int = 1) -> None:
        self.excluded[component] = self.excluded.get(component, 0) + n

    def deny(self, reason: str) -> "Projection":
        self.allowed = False
        self.reason = reason
        self.messages = None
        return self

    def as_dict(self) -> Dict[str, Any]:
        d = {"allowed": self.allowed, "excluded_components": dict(self.excluded)}
        if not self.allowed:
            d["reason"] = self.reason
        return d


def _filter_skills_index(text: str, match) -> Tuple[str, int]:
    """Drop whole skill-index entries whose name or description matches policy.

    Operates on the entries Hermes renders inside ``<available_skills>``:
    ``    - name: description`` lines, and ``  category [names only]: a, b, c``
    lines. Whole entries go, never a substring of one. The skill files on disk
    are not touched and local sessions still see every skill.
    """
    removed = 0

    def _block(m: re.Match) -> str:
        nonlocal removed
        kept: List[str] = []
        dropping_category = False
        for line in m.group(2).split("\n"):
            entry = line.strip()
            if entry.startswith("- "):
                # A dropped category takes its indented entries with it,
                # otherwise they get reparented under the previous category and
                # the index misrepresents what is available.
                if dropping_category or match(entry):
                    removed += 1
                    continue
            elif "[names only]:" in entry:
                dropping_category = False
                head, _, names = entry.partition("[names only]:")
                keep_names = []
                for name in names.split(","):
                    if name.strip() and match(name.strip()):
                        removed += 1
                        continue
                    if name.strip():
                        keep_names.append(name.strip())
                if not keep_names:
                    continue
                indent = line[: len(line) - len(line.lstrip())]
                line = f"{indent}{head}[names only]: {', '.join(keep_names)}"
            elif entry.endswith(":") or (entry and not entry.startswith("- ")):
                # Category header line. If it matches, drop it and everything
                # indented beneath it.
                dropping_category = bool(match(entry))
                if dropping_category:
                    removed += 1
                    continue
            kept.append(line)
        return m.group(1) + "\n".join(kept) + m.group(3)

    return _SKILLS_BLOCK_RE.sub(_block, text), removed


def _filter_memory(text: str, match) -> Tuple[str, int]:
    """Drop whole memory / user-profile entries that match policy.

    Entries are the ``§``-delimited records Hermes stores. A matching entry is
    removed in full — never partially rewritten — and the memory files on disk
    are not modified.
    """
    removed = 0

    def _block(m: re.Match) -> str:
        nonlocal removed
        entries = m.group(2).split(_MEM_ENTRY_DELIM)
        kept = []
        for e in entries:
            if e.strip() and match(e):
                removed += 1
                continue
            kept.append(e)
        return m.group(1) + _MEM_ENTRY_DELIM.join(kept)

    return _MEM_BLOCK_RE.sub(_block, text), removed


def _project_system(content: str, match, proj: Projection) -> Optional[str]:
    """Project one system message. Returns None when it cannot be made safe."""
    if not match(content):
        return content

    # Only filter framing that Hermes itself emitted. A bare "<available_skills>"
    # or a hand-made separator row can appear in a user's own context file, and
    # silently deleting a line out of THAT would be sanitizing user content.
    # Without the canonical preamble the block is unattributable, so the
    # request refuses below instead.
    text = content
    if _FRAMEWORK_SKILLS_PREAMBLE in text:
        text, n = _filter_skills_index(text, match)
        if n:
            proj._exclude("skills_index", n)
    text, n = _filter_memory(text, match)
    if n:
        proj._exclude("memory", n)

    if match(text):
        # Structured filtering could not attribute the remaining match to a
        # block we know how to remove — a context file, a plugin-added block, a
        # new prompt component. Refuse rather than guess.
        return None
    return text


def project_messages(messages: List[Dict[str, Any]]) -> Projection:
    """Produce an external-safe projection of an outbound message list.

    Framework contamination is removed structurally. User content and
    non-framework tool results are never rewritten: if they carry protected
    material the projection is refused and the caller must not send the request.
    """
    proj = Projection()
    try:
        match = _matcher()
    except PolicyUnavailable as exc:
        proj.policy_reason = str(exc)
        return proj.deny(f"policy_unavailable:{exc}")

    out: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        role = msg.get("role")
        content = msg.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False) \
            if content is not None else ""

        if role == "system":
            safe = _project_system(text, match, proj)
            if safe is None:
                return proj.deny("protected_context_remaining")
            new = dict(msg)
            if isinstance(content, str):
                new["content"] = safe
            elif safe != text:
                # Non-string system content that needed filtering cannot be
                # rebuilt faithfully — refuse instead of risking a malformed body.
                return proj.deny("unstructured_system_content")
            out.append(new)
            continue

        if role == "tool":
            if not match(text):
                out.append(msg)
                continue
            # Framework skill content may be replaced wholesale. Anything else
            # is workspace data and must not be sanitized into passing.
            tool_name = str(msg.get("name") or "")
            if tool_name in FRAMEWORK_SKILL_TOOLS:
                new = dict(msg)
                new["content"] = SKILL_BODY_MARKER
                proj._exclude("skill_body", 1)
                out.append(new)
                continue
            return proj.deny("protected_tool_result")

        # user / assistant / anything else: never sanitized.
        if match(text):
            return proj.deny("protected_user_content" if role == "user"
                             else "protected_message_content")
        out.append(msg)

    proj.messages = out
    return proj


# ---------------------------------------------------------------- final gate


class UnscannablePayload(Exception):
    """Raised when a payload cannot be traversed exhaustively.

    The caller turns this into a REFUSAL. A scan that cannot see all of the
    content must never report the content clean.
    """


#: Upper bound on nodes visited in one traversal. Not a depth limit — a total
#: work limit, so a pathological payload cannot hang the request path. Hitting
#: it raises rather than truncating.
_SCAN_NODE_BUDGET = 2_000_000


def _iter_strings(node: Any):
    """Yield every raw string leaf, and every dict key, in a request payload.

    Iterative on an explicit stack, with NO depth cap. The previous recursive
    version returned silently at depth 40, so anything nested below that escaped
    this scan entirely — and because the sibling blob scan cannot see through
    JSON escaping, a whitespace-split identifier nested past 40 levels escaped
    both gates. A cap that silently stops scanning is indistinguishable from a
    clean result, which is the wrong failure direction for a security gate.

    Cycles are tolerated (each container is visited once) because a cyclic
    payload is unserializable and is refused by the blob scan anyway. Anything
    that cannot be traversed raises UnscannablePayload.
    """
    stack = [node]
    seen: set = set()
    # `seen` holds id() values, which CPython reuses once an object is freed.
    # Every container reachable from the root stays alive for the whole walk, so
    # reuse cannot happen today — but a freed-then-reused id would silently mark
    # an unvisited container as already scanned, i.e. fail open. Pin them.
    keepalive: list = []
    budget = _SCAN_NODE_BUDGET
    while stack:
        budget -= 1
        if budget <= 0:
            raise UnscannablePayload("payload exceeds the scan budget")
        cur = stack.pop()
        if isinstance(cur, str):
            yield cur
            continue
        if isinstance(cur, (int, float, bool)) or cur is None:
            continue                # scalars cannot carry an identifier
        if isinstance(cur, (dict, list, tuple, set, frozenset)):
            marker = id(cur)
            if marker in seen:
                continue            # already traversed (cycle or shared node)
            seen.add(marker)
            keepalive.append(cur)
        if isinstance(cur, dict):
            try:
                items = list(cur.items())
            except Exception as exc:
                raise UnscannablePayload("dict could not be enumerated") from exc
            for k, v in items:
                if isinstance(k, str):
                    yield k
                elif not (isinstance(k, (int, float, bool)) or k is None):
                    stack.append(k)
                stack.append(v)
            continue
        if isinstance(cur, (list, tuple, set, frozenset)):
            try:
                stack.extend(cur)
            except Exception as exc:
                raise UnscannablePayload("sequence could not be enumerated") from exc
            continue
        _iter_object_fallback(cur, stack)


def _iter_object_fallback(cur: Any, stack: list) -> None:
    """An arbitrary object reaches the wire through json.dumps(default=str), so
    its str() form is outbound content. If it cannot be stringified we cannot
    know what would be sent, so refuse rather than skip it."""
    try:
        stack.append(str(cur))
    except Exception as exc:
        raise UnscannablePayload("object is not stringifiable") from exc


def final_scan(payload: Any) -> Tuple[bool, str]:
    """Mandatory gate over the outbound request, immediately before the wire.

    Scans in two complementary ways, because neither alone is sufficient:

    * **Every raw string leaf and dict key.** This is the authoritative pass.
      It must NOT be replaced by a scan of the serialized form: ``json.dumps``
      rewrites a real newline, tab, CR, form-feed or vertical-tab as the two
      characters ``\\`` + ``n``, and the guard's ``normalize()`` collapses runs
      of *real* whitespace to ``-``. Scanning only the serialized blob therefore
      MISSES an identifier broken across a line — which is ordinary in wrapped
      prose, log output and tables — while the guard matches it happily in the
      raw string. Scanning the raw leaves keeps this gate exactly as strong as
      the guard it delegates to.
    * **The serialized blob**, additionally, to catch what per-leaf scanning
      cannot: accidental concatenation across fields and structural oddities.

    The gateway performs its own independent DLP scan afterwards; this one only
    decides whether Hermes is willing to put the request on the wire at all.
    """
    try:
        match = _matcher()
    except PolicyUnavailable as exc:
        return False, f"policy_unavailable:{exc}"

    try:
        # Every raw leaf, scanned in ONE pass. Leaves are joined with NUL: no
        # policy pattern can match across it (they are built from [a-z0-9-]
        # classes and literals), so this cannot merge two innocent leaves into
        # a false match, and it avoids re-invoking the matcher per leaf.
        joined = "\x00".join(x for x in _iter_strings(payload) if x)
    except UnscannablePayload as exc:
        return False, f"unscannable_payload:{exc}"
    except Exception:
        return False, "unscannable_payload"
    if joined and match(joined):
        return False, "protected_match_in_request"

    try:
        blob = payload if isinstance(payload, str) else json.dumps(
            payload, ensure_ascii=False, default=str)
    except Exception:
        return False, "unserializable_payload"
    if match(blob):
        # The matched rule is a policy pattern rather than user content, but it
        # is not needed downstream and is deliberately not returned.
        return False, "protected_match_in_serialized_request"
    return True, "ok"


# ---------------------------------------------------------------- audit


def audit(request_id: str, decision: str, reason: str,
          excluded: Optional[Dict[str, int]] = None) -> None:
    """Append one metadata-only record. Never writes prompt or removed content."""
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request_id": request_id,
        "decision": decision,
        "reason": reason,
        "excluded_counts": excluded or {},
    }
    try:
        path = os.path.expanduser("~/.hermes/logs/external_projection.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        with open(fd, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        # Auditing must never break the request path.
        pass


# ---------------------------------------------------------------- entry point


def guard_outbound(api_kwargs: Dict[str, Any], *, provider: Optional[str],
                   base_url: Optional[str], request_id: str = "") -> Dict[str, Any]:
    """Project and gate one outbound request.

    Returns ``{"allowed": True, "api_kwargs": ...}`` for a request Hermes is
    willing to send, or ``{"allowed": False, "reason": ...}`` when it must be
    refused locally. Local targets are returned untouched.
    """
    if not is_external_target(provider, base_url):
        return {"allowed": True, "api_kwargs": api_kwargs, "external": False}

    # NOTE: the codex path gates twice (the streaming choke point delegates to
    # the non-streaming one). That is idempotent — the second pass finds nothing
    # to filter and returns the same object — but it does count the request
    # twice in the projection metrics. Deliberately not deduplicated with a
    # marker on the payload: any extra key here would be forwarded into
    # chat.completions.create(**api_kwargs) as an unknown parameter.

    _count("elite_hermes_external_projection_total", decision="attempted")

    # Fast path: scan the fully serialized request first. When nothing matches
    # there is nothing to filter, and this single pass IS the mandatory final
    # gate — the request goes out having been scanned in full, exactly once.
    ok, reason = final_scan(api_kwargs)
    if ok:
        _count("elite_hermes_external_projection_total", decision="allowed")
        audit(request_id, "allowed", "ok", {})
        return {"allowed": True, "api_kwargs": api_kwargs, "external": True,
                "excluded_components": {}}
    if reason.startswith("policy_unavailable") or reason == "unserializable_payload":
        _count("elite_hermes_external_projection_block_total", reason=reason)
        audit(request_id, "blocked", reason, {})
        return {"allowed": False, "reason": reason, "external": True}

    # Something matched. Attribute it to a structured block and remove only what
    # is framework-owned; anything else refuses below.
    messages = api_kwargs.get("messages")
    projected = dict(api_kwargs)
    excluded: Dict[str, int] = {}

    # anthropic_messages and bedrock_converse carry the system prompt as a
    # top-level kwarg rather than a role=="system" message, so it has to be
    # projected here or framework contamination in those modes is unfilterable
    # and every such request is refused.
    sys_kw = api_kwargs.get("system")
    if sys_kw is not None:
        try:
            match = _matcher()
        except PolicyUnavailable as exc:
            _count("elite_hermes_external_projection_block_total",
                   reason="policy_unavailable")
            audit(request_id, "blocked", f"policy_unavailable:{exc}", {})
            return {"allowed": False, "reason": f"policy_unavailable:{exc}",
                    "external": True}
        sub = Projection()
        if isinstance(sys_kw, str):
            safe = _project_system(sys_kw, match, sub)
            if safe is None:
                _count("elite_hermes_external_projection_block_total",
                       reason="protected_system_kwarg")
                audit(request_id, "blocked", "protected_system_kwarg", sub.excluded)
                return {"allowed": False, "reason": "protected_system_kwarg",
                        "external": True}
            projected["system"] = safe
        elif isinstance(sys_kw, list):
            out_blocks = []
            for blk in sys_kw:
                if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                    safe = _project_system(blk["text"], match, sub)
                    if safe is None:
                        _count("elite_hermes_external_projection_block_total",
                               reason="protected_system_kwarg")
                        audit(request_id, "blocked", "protected_system_kwarg",
                              sub.excluded)
                        return {"allowed": False, "reason": "protected_system_kwarg",
                                "external": True}
                    nb = dict(blk)
                    nb["text"] = safe
                    out_blocks.append(nb)
                else:
                    out_blocks.append(blk)
            projected["system"] = out_blocks
        for c, n in sub.excluded.items():
            excluded[c] = excluded.get(c, 0) + n

    if isinstance(messages, list):
        proj = project_messages(messages)
        if not proj.allowed:
            _count("elite_hermes_external_projection_block_total", reason=proj.reason)
            audit(request_id, "blocked", proj.reason, proj.excluded)
            return {"allowed": False, "reason": proj.reason, "external": True}
        projected["messages"] = proj.messages
        # Merge, never assign: the top-level ``system`` kwarg may already have
        # contributed exclusions above, and overwriting them under-reports what
        # was removed.
        for _c, _n in proj.excluded.items():
            excluded[_c] = excluded.get(_c, 0) + _n
        for component in excluded:
            _count("elite_hermes_external_component_excluded_total",
                   component_type=component)

    # Mandatory gate over the whole serialized request, re-run after filtering.
    ok, reason = final_scan(projected)
    if not ok:
        _count("elite_hermes_external_projection_block_total", reason=reason)
        audit(request_id, "blocked", reason, excluded)
        return {"allowed": False, "reason": reason, "external": True}

    _count("elite_hermes_external_projection_total", decision="allowed")
    audit(request_id, "allowed", "ok", excluded)
    return {"allowed": True, "api_kwargs": projected, "external": True,
            "excluded_components": excluded}


__all__ = [
    "PolicyUnavailable", "ExternalProjectionBlocked", "Projection", "policy_state", "is_external_target",
    "project_messages", "final_scan", "guard_outbound", "audit", "metrics_text",
    "COUNTERS", "FRAMEWORK_SKILL_TOOLS", "SKILL_BODY_MARKER",
]
