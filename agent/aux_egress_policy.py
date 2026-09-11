"""Egress policy for the Hermes auxiliary/background LLM stack.

Auxiliary features — context and trajectory compression, the curator, title
generation, insights, session search, vision, goals and kanban helpers — build
their own outbound LLM requests through :mod:`agent.auxiliary_client` rather
than through the main agent loop. That resolver ends in an ambient-credential
discovery step: it walks a provider registry and returns a client pointed at
whichever public provider happens to have a key in the environment.

That path bypasses everything the Elite external gateway exists to enforce —
DLP, project authorization, budgets, rate limits, audit, circuit breakers — and
it is reachable in ordinary operation, not just in theory: an auxiliary request
to the gateway that fails auth is classified as a fallback-worthy error, and the
fallback chain then resolves to a public provider.

This module is the choke point that makes that impossible. Every auxiliary
client construction is classified against endpoints that are **explicitly
approved in configuration**:

* the sanctioned Elite external AI gateway — the only route off this machine;
* approved local inference routes — LiteLLM and any route listed in config.

Locality is deliberately NOT inferred from an address being private. A private
or Tailscale address is not evidence that a service is an approved local model:
a forwarded port or an unrelated host on the tailnet would otherwise be treated
as trusted. Only configured, named routes count.

Anything else — and anything that cannot be classified — is refused. There is no
"probably fine" branch and no fall-through to a public provider.

Proxies are a separate, second decision. An approved destination reached through
a proxy is not an approved request: the proxy terminates the connection, so it —
not the destination — receives the gateway credential and the whole prompt body.
Being approved to RECEIVE inference therefore does not make an endpoint approved
to RELAY it, and ``approved_local_routes()`` is deliberately not the proxy
allowlist. Absent an explicitly sanctioned forward proxy, any real proxy in the
path fails closed. ``NO_PROXY`` is bypass metadata on the other side of that
question and is never treated as a proxy destination.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: The one sanctioned route off this machine. Pinned rather than configurable:
#: the whole point is that auxiliary code cannot elect a different egress.
SANCTIONED_GATEWAY_HOST = "100.100.175.79"
SANCTIONED_GATEWAY_PORT = "8082"

#: Environment variable holding the gateway credential. Auxiliary requests to
#: the gateway must authenticate or they 401 — and a 401 is precisely what the
#: auxiliary fallback chain treats as licence to try a public provider instead.
GATEWAY_KEY_ENV = "ELITE_GATEWAY_KEY"

#: Provider hostnames that must never be reached directly from this machine.
#: Used for reporting and for the canary assertion in tests; enforcement is by
#: allowlist, so a provider missing from this list is still refused.
KNOWN_PUBLIC_PROVIDER_HOSTS = (
    "api.openai.com", "openai.azure.com", "api.anthropic.com", "api.deepseek.com",
    "openrouter.ai", "generativelanguage.googleapis.com", "api.z.ai",
    "open.bigmodel.cn", "api.x.ai", "api.groq.com", "api.mistral.ai",
    "api.kimi.com", "integrate.api.nvidia.com", "inference-api.nousresearch.com",
    "api.together.xyz", "api.fireworks.ai", "bedrock-runtime",
    "api.stepfun.com", "api.arcee.ai", "githubcopilot.com", "chatgpt.com",
)

GATEWAY = "gateway"
LOCAL = "local"
FORBIDDEN = "forbidden"


class AuxEgressBlocked(Exception):
    """Raised instead of building an auxiliary client to an unapproved endpoint.

    Carries ``status_code = 400`` so that, if it ever surfaces through Hermes'
    error classifier, it reads as a non-retryable client error rather than an
    auth failure — an auth classification would drive credential rotation for
    what is really a local policy decision.
    """

    status_code = 400

    def __init__(self, host: str, where: str, reason: str) -> None:
        self.host = host
        self.where = where
        self.reason = reason
        super().__init__(
            f"auxiliary egress to '{host}' refused by Elite policy at {where} "
            f"({reason}). Auxiliary LLM work must go through the sanctioned "
            f"Elite gateway or an approved local route. No request was sent."
        )


# ------------------------------------------------------------------ config


_config_cache: Dict[str, Any] = {}


def _config() -> dict:
    """Load Hermes config, cached on the config file's identity.

    classify() runs on every client construction and every credential-pool
    seed, and each call previously re-read and re-parsed the whole config.
    """
    try:
        from hermes_cli.config import load_config, get_config_path
        try:
            path = os.path.realpath(get_config_path())
            st = os.stat(path)
            # Include the resolved path and inode: a stat-only key serves a
            # stale parse when a file is rewritten to the same size with the
            # mtime restored, and serves the wrong entry across a HERMES_HOME
            # swap between stat-identical configs.
            stamp = (path, st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)
        except Exception:
            stamp = None
        if stamp is not None and _config_cache.get("stamp") == stamp:
            return _config_cache["cfg"]
        cfg = load_config()
        cfg = cfg if isinstance(cfg, dict) else {}
        if stamp is not None:
            _config_cache["stamp"] = stamp
            _config_cache["cfg"] = cfg
        return cfg
    except Exception:
        return {}


#: Characters that make an authority ambiguous between parsers. A URL carrying
#: any of them is refused outright rather than guessed at.
_AMBIGUOUS_AUTHORITY = ("@", "\\", "#", "?", " ", "\t", "\n", "\r", "\x00")


def _host_port(url: str) -> Tuple[str, str]:
    """Split a URL into (host, port) using the SAME semantics as the HTTP client.

    This must not be a hand-rolled split. The previous implementation partitioned
    on the first ":" and never stripped RFC-3986 userinfo, so
    ``http://127.0.0.1:11434@evil.example.com/v1`` read as host ``127.0.0.1``
    while httpx resolves ``evil.example.com``. Since the allowlist builder and
    the classifier share this function, such a URL in config registered a route
    that then matched itself and classified LOCAL — a full bypass from an input
    this module treats as untrusted.

    Returns ("", "") when the URL is unparseable or its authority is ambiguous,
    which classify() then treats as FORBIDDEN.
    """
    if not isinstance(url, str):
        return "", ""
    u = url.strip()
    if not u:
        return "", ""

    # Authority = everything after the scheme, before the first '/'. Reject
    # ambiguity here rather than letting two parsers disagree about it.
    probe = re.sub(r"^[A-Za-z0-9+.\-]+://", "", u, count=1)
    authority = probe.split("/", 1)[0]
    if any(c in authority for c in _AMBIGUOUS_AUTHORITY):
        return "", ""

    try:
        from urllib.parse import urlsplit
        parts = urlsplit(u if "://" in u else "//" + u)
        host = (parts.hostname or "").lower()
        port = str(parts.port) if parts.port is not None else ""
    except ValueError:
        return "", ""
    return host, port


def _is_plausibly_local(host: str) -> bool:
    """Whether a host could be estate-internal at all.

    This is a NECESSARY condition layered under the configured-route check, not
    a sufficient one: a private address still has to appear in configuration to
    be approved. It exists only to stop a public hostname that someone added to
    ``providers:`` from being promoted to "approved local route".
    """
    h = (host or "").strip("[]").lower()
    if not h:
        return False
    if h in ("localhost", "::1"):
        return True
    try:
        import ipaddress
        ip = ipaddress.ip_address(h)
        # Link-local is deliberately NOT local: 169.254.169.254 is the cloud
        # metadata service, and treating it as an approved route makes instance
        # credentials reachable from the auxiliary stack.
        if ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            return False
        if ip.is_reserved:
            return False
        # 6to4 (2002::/16) and teredo (2001::/23) report as "private" to
        # ipaddress but EMBED an arbitrary IPv4 — including the metadata
        # service and public addresses — so they are not evidence of locality.
        if ip.version == 6:
            for embedded in ("2002::/16", "2001::/23"):
                if ip in ipaddress.ip_network(embedded):
                    return False
        if ip.is_private or ip.is_loopback:
            return True
        # Tailscale / CGNAT shared address space, where the Elite cluster lives.
        return ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        pass
    # A single-label name from CONFIG is not accepted: the config path resolves
    # through HERMES_HOME, so a writable config could name "exfil-host", which
    # the DNS search domain then resolves to anywhere. Only literal
    # private/loopback/CGNAT addresses count here. ("localhost" is still
    # reachable — it is matched by the explicit check above, not by this rule.)
    return False


def approved_local_routes() -> List[Tuple[str, str]]:
    """(host, port) pairs for local inference routes named in configuration.

    Read from the configured providers rather than assumed, so adding a local
    route is a configuration change and not a matter of an address looking
    private. The sanctioned gateway is excluded here — it is external egress
    and is classified separately.
    """
    routes: List[Tuple[str, str]] = []
    cfg = _config()
    providers = dict(cfg.get("providers") or {})
    # Hermes also supports a legacy `custom_providers:` list. A route configured
    # that way is just as approved as one under `providers:`; reading only the
    # latter refused it.
    legacy = cfg.get("custom_providers")
    if isinstance(legacy, list):
        for i, spec in enumerate(legacy):
            if isinstance(spec, dict):
                providers[f"__custom_{i}"] = spec
    elif isinstance(legacy, dict):
        providers.update({k: v for k, v in legacy.items() if isinstance(v, dict)})
    if isinstance(providers, dict):
        for name, spec in providers.items():
            if not isinstance(spec, dict):
                continue
            url = spec.get("base_url") or spec.get("api_base") or ""
            host, port = _host_port(str(url))
            if not host:
                continue
            if (host, port) == (SANCTIONED_GATEWAY_HOST, SANCTIONED_GATEWAY_PORT):
                continue                              # that is the gateway
            # Being listed under `providers:` does NOT make an endpoint local.
            # The same map legitimately contains public providers (openai,
            # anthropic, …); treating those as approved local routes would hand
            # back exactly the direct egress this module exists to remove.
            if any(pub in host for pub in KNOWN_PUBLIC_PROVIDER_HOSTS):
                continue
            if not _is_plausibly_local(host):
                continue
            if not port:
                # A config route with no port would approve the whole host.
                # Require the service to be named explicitly.
                continue
            routes.append((host, port))

    # Loopback inference servers that ship with Hermes' own defaults.
    # NOTE: no bracketed "[::1]" form — _host_port parses with urlsplit,
    # whose .hostname strips brackets, so a host is never returned bracketed.
    for h in ("127.0.0.1", "localhost", "::1"):
        routes.extend([(h, "11434"), (h, "1234")])
    return routes


def is_sanctioned_gateway(url: str) -> bool:
    host, port = _host_port(url)
    return host == SANCTIONED_GATEWAY_HOST and port == SANCTIONED_GATEWAY_PORT


# ------------------------------------------------------------------ policy


#: Enforcement is UNCONDITIONAL in production. There is deliberately no
#: configuration key and no environment variable that turns it off.
#:
#: An earlier version honoured `auxiliary.elite_egress_lockdown: false` from the
#: config file. That was a one-variable bypass of the entire lockdown: the config
#: path is resolved through HERMES_HOME, so pointing that at a directory with a
#: two-line config.yaml disabled enforcement without touching the real config —
#: and the config file is writable by Hermes' own file tools. A kill switch
#: reachable from ambient process state is exactly the class of input this
#: module exists to stop trusting.
#:
#: Tests disable it by patching this module attribute in-process, which requires
#: code execution inside Hermes rather than a writable environment.
_ENFORCE = True


def enabled() -> bool:
    """Whether the lockdown is active. True in production, always."""
    return bool(_ENFORCE)


def classify(url: Optional[str]) -> Tuple[str, str]:
    """Classify an auxiliary egress target. Returns (verdict, reason)."""
    host, port = _host_port(url or "")
    if not host:
        return FORBIDDEN, "no resolvable host"
    if host == SANCTIONED_GATEWAY_HOST and port == SANCTIONED_GATEWAY_PORT:
        return GATEWAY, "sanctioned Elite external AI gateway"
    # NOTE: the "gateway host on unapproved port" branch below deliberately runs
    # AFTER the configured routes. Hoisting it above them looks tighter but
    # breaks the real deployment: LiteLLM is an approved local route on the SAME
    # host as the gateway (100.100.175.79:4000), so an unconditional check here
    # refuses it. A config entry naming another port on that host is exactly as
    # trusted as any other config route, and is already bounded by
    # _is_plausibly_local plus the explicit-port requirement.
    for ahost, aport in approved_local_routes():
        # Exact host AND port. A configured route with no port must not
        # approve every port on that host — that silently widens one named
        # service into the whole machine.
        if host == ahost and port == aport:
            return LOCAL, "approved local inference route"
    if any(p in host for p in KNOWN_PUBLIC_PROVIDER_HOSTS):
        return FORBIDDEN, "direct public provider"
    if host == SANCTIONED_GATEWAY_HOST:
        return FORBIDDEN, f"gateway host on unapproved port {port or '(none)'}"
    return FORBIDDEN, "not an approved route"



# ------------------------------------------------------------------ proxies

#: The only proxy keys an HTTP transport can actually act on.
#:
#: urllib's ``getproxies_environment()`` turns EVERY ``*_proxy`` variable into a
#: key, so ``NO_PROXY`` arrives as ``no`` and ``TRAVIS_APT_PROXY`` as
#: ``travis_apt``. Iterating those values as if they were proxy URLs was wrong
#: in both directions: ``NO_PROXY=localhost,127.0.0.1`` — a BYPASS list, the
#: statement that certain hosts are reached directly — was parsed as a proxy
#: destination, classified FORBIDDEN, and blocked ordinary gateway traffic; and
#: unrelated tooling variables were reported as transport proxies they are not.
#:
#: httpx mounts only these three keys and deliberately drops the rest ("we don't
#: want to propagate non-HTTP proxies into our configuration such as
#: 'TRAVIS_APT_PROXY'"), and every other trust_env client behaves the same way.
#: Mixed-case variables stay covered because getproxies_environment() lower-cases
#: each NAME before matching, so ``Http_Proxy`` still arrives as key ``http``.
_TRANSPORT_PROXY_KEYS = ("http", "https", "all")


def _proxy_env_map() -> Dict[str, str]:
    """The proxy map the clients themselves resolve, keys lower-cased.

    getproxies(), NOT getproxies_environment(): every trust_env client resolves
    through getproxies(), which falls back to the PLATFORM proxy (macOS sysconf,
    Windows registry) when no variable is set. Reading only the environment left
    that channel invisible to the policy while the client honoured it — and the
    Windows registry key is user-writable with no privilege.
    """
    try:
        import urllib.request
        raw = urllib.request.getproxies() or {}
    except Exception:
        return {}
    out: Dict[str, str] = {}
    for key, val in raw.items():
        try:
            out[str(key).lower()] = str(val or "").strip()
        except Exception:
            continue
    return out


def _is_ip_literal(value: str) -> bool:
    try:
        import ipaddress
        ipaddress.ip_address(value.strip("[]"))
        return True
    except Exception:
        return False


def no_proxy_bypasses(host: str, no_proxy: Optional[str] = None) -> bool:
    """Whether ``NO_PROXY`` says ``host`` is reached directly.

    Matching follows the rules the transports use (curl's CURLOPT_NOPROXY, which
    httpx mirrors): ``*`` disables all proxies; a literal IP or ``localhost``
    matches exactly; a domain matches itself and its subdomains, so
    ``google.com`` covers ``www.google.com`` but not ``wwwgoogle.com``; and CIDR
    entries match by network membership.

    This is the *bypass* question only. NO_PROXY is never a proxy destination —
    conflating the two is the bug this function exists to make impossible.
    """
    h = (host or "").strip().strip("[]").lower().rstrip(".")
    if not h:
        return False
    if no_proxy is None:
        no_proxy = _proxy_env_map().get("no", "")
    for entry in str(no_proxy or "").split(","):
        e = entry.strip().lower()
        if not e:
            continue
        if e == "*":
            return True
        if "://" in e:                                 # NO_PROXY=http://host
            e = e.split("://", 1)[1]
        if "/" in e:                                   # CIDR, e.g. 192.168.0.0/16
            try:
                import ipaddress
                if ipaddress.ip_address(h) in ipaddress.ip_network(e, strict=False):
                    return True
            except Exception:
                pass
            continue
        head, sep, tail = e.rpartition(":")            # trailing :port
        if sep and head and tail.isdigit() and not head.endswith(":"):
            e = head
        e = e.strip("[]").rstrip(".")
        if not e:
            continue
        if h == e:
            return True
        if _is_ip_literal(e) or e == "localhost":
            continue                                   # exact match only
        if h.endswith("." + e.lstrip(".")):
            return True
    return False


def candidate_proxies(url: str) -> List[str]:
    """EVERY proxy that could carry ``url``, not just the first-wins pick.

    Clients differ in which variable they honour: ones built through Hermes'
    keepalive helper receive an explicit proxy, while the Anthropic SDK and a
    bare httpx client (``trust_env=True``) make their own scheme-correct choice.
    Classifying only the first-wins value let an approved decoy in HTTPS_PROXY
    hide an attacker value in HTTP_PROXY for a plain-HTTP destination, so every
    proxy any transport in use would mount is returned.

    Each transport is asked under ITS OWN bypass rule, so the result is what
    would really happen rather than a re-derivation that can drift from it.
    """
    host, _ = _host_port(url)
    out: List[str] = []

    def _add(value: Any) -> None:
        val = str(value or "").strip()
        if val and val not in out:
            out.append(val)

    # 1. Hermes' own keepalive client. It reads the exact-case HTTPS/HTTP/ALL
    #    variables and applies urllib's NO_PROXY rule itself, so it is asked as
    #    a transport rather than modelled here.
    try:
        from agent.process_bootstrap import _get_proxy_for_base_url
        _add(_get_proxy_for_base_url(url))
    except Exception:
        pass

    # 2. What a trust_env httpx / SDK client would mount, including the platform
    #    proxy. Skipped entirely when NO_PROXY covers the destination — that is
    #    the transport reaching it directly, not a proxy to classify.
    env = _proxy_env_map()
    if not (host and no_proxy_bypasses(host, env.get("no", ""))):
        for key in _TRANSPORT_PROXY_KEYS:
            _add(env.get(key))
    return out


def effective_proxy(url: str) -> Optional[str]:
    """The first proxy that would carry ``url``, or None. See candidate_proxies."""
    got = candidate_proxies(url)
    return got[0] if got else None


def approved_proxies() -> List[Tuple[str, str]]:
    """(host, port) pairs explicitly approved to act as a FORWARD PROXY.

    Deliberately NOT ``approved_local_routes()``. Those two lists answer
    different questions, and reusing one for the other was a real hole: an
    endpoint approved to RECEIVE an inference request is not thereby approved to
    CARRY one. With the old rule ``HTTP_PROXY=http://127.0.0.1:11434`` was
    accepted — because Ollama is an approved local model — and every auxiliary
    request, the ELITE_GATEWAY_KEY and the full prompt body with it, was handed
    in cleartext to whatever was listening on that port. Same for LiteLLM on
    :4000. A forward proxy sees the credential and the content of every request
    it relays; an inference endpoint sees only what is addressed to it.

    Default is EMPTY, so an actual proxy in the path fails closed. Entries are
    bounded the same way local routes are — plausibly-local host, explicit port,
    never a known public provider — because this config path resolves through
    HERMES_HOME and is writable by Hermes' own file tools; an unbounded proxy
    allowlist there would be a one-line exfiltration switch. A sanctioned proxy
    off this machine would have to be pinned in code, like the gateway itself.
    """
    out: List[Tuple[str, str]] = []
    section = _config().get("auxiliary")
    raw: List[Any] = []
    if isinstance(section, dict):
        val = section.get("approved_egress_proxies")
        if isinstance(val, str):
            raw = [val]
        elif isinstance(val, (list, tuple)):
            raw = list(val)
    for item in raw:
        host, port = _host_port(str(item or ""))
        if not host or not port:
            continue
        if any(pub in host for pub in KNOWN_PUBLIC_PROVIDER_HOSTS):
            continue
        if not _is_plausibly_local(host):
            continue
        if (host, port) not in out:
            out.append((host, port))
    return out


def proxy_is_approved(url: str) -> Tuple[bool, str]:
    """Is the proxy for ``url`` — if there is one — approved to carry it?

    Classifying only the destination is insufficient. HTTP_PROXY/HTTPS_PROXY
    redirect the connection regardless of what the base_url says, and gateway
    traffic is plain HTTP, so an unapproved proxy would receive the gateway
    credential and the entire prompt body in cleartext. That would defeat the
    DLP gateway without touching anything else this module gates.
    """
    allowed = approved_proxies()
    for proxy in candidate_proxies(url):
        phost, pport = _host_port(proxy)
        if not phost:
            return False, "traffic could be proxied via an unparseable proxy URL"
        if (phost, pport) not in allowed:
            return False, (f"traffic could be proxied via unapproved forward "
                           f"proxy {phost}:{pport or '(none)'}")
    return True, "no unapproved proxy"


def assert_allowed(url: Optional[str], where: str) -> str:
    """Raise unless ``url`` is an approved auxiliary egress target."""
    if not enabled():
        return "disabled"
    verdict, reason = classify(url)
    if verdict == FORBIDDEN:
        host, _ = _host_port(url or "")
        logger.warning("auxiliary egress refused at %s: host=%s reason=%s",
                       where, host or "<none>", reason)
        _count(where, reason)
        raise AuxEgressBlocked(host or "<none>", where, reason)

    # An approved destination reached through an unapproved proxy is not an
    # approved request: the proxy, not the destination, is where the bytes go.
    ok, proxy_reason = proxy_is_approved(url or "")
    if not ok:
        host, _ = _host_port(url or "")
        logger.warning("auxiliary egress refused at %s: host=%s reason=%s",
                       where, host or "<none>", proxy_reason)
        _count(where, "proxied via unapproved host")
        raise AuxEgressBlocked(host or "<none>", where, proxy_reason)
    return verdict


def gateway_credential() -> str:
    """The sanctioned gateway credential, or "" when it is not available.

    Without this an auxiliary request to the gateway is sent unauthenticated and
    401s — and the auxiliary fallback chain reads a 401 as permission to try a
    public provider instead. Supplying it is therefore part of closing the
    bypass, not a convenience.
    """
    val = (os.environ.get(GATEWAY_KEY_ENV) or "").strip()
    if val:
        return val
    # Hermes loads ~/.hermes/.env into the process, but auxiliary code can run
    # before that happens; read it directly rather than fail auth.
    try:
        path = os.path.expanduser("~/.hermes/.env")
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = re.match(r"^\s*(?:export\s+)?" + GATEWAY_KEY_ENV
                             + r"\s*=\s*(.+?)\s*$", line)
                if m:
                    return m.group(1).strip().strip("\"'")
    except Exception:
        pass
    return ""


# ------------------------------------------------------------------ metrics

#: Bounded counters: the label set is (where, reason) drawn from closed
#: vocabularies in this module. No URL, credential, prompt or content.
COUNTERS: Dict[Tuple[str, str], int] = {}


def _count(where: str, reason: str) -> None:
    key = (where, reason)
    COUNTERS[key] = COUNTERS.get(key, 0) + 1


def metrics_text() -> List[str]:
    return [f'elite_aux_egress_blocked_total{{site="{w}",reason="{r}"}} {n}'
            for (w, r), n in sorted(COUNTERS.items())]


__all__ = [
    "AuxEgressBlocked", "candidate_proxies", "proxy_is_approved", "approved_proxies",
    "no_proxy_bypasses", "effective_proxy", "GATEWAY", "LOCAL", "FORBIDDEN",
    "SANCTIONED_GATEWAY_HOST", "SANCTIONED_GATEWAY_PORT", "GATEWAY_KEY_ENV",
    "KNOWN_PUBLIC_PROVIDER_HOSTS", "classify", "assert_allowed", "enabled",
    "approved_local_routes", "is_sanctioned_gateway", "gateway_credential",
    "metrics_text", "COUNTERS",
]
