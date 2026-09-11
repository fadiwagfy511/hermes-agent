"""Tests for the Elite auxiliary-stack egress lockdown.

The auxiliary/background LLM stack (compression, title generation, vision, MoA,
kanban/goals helpers, MCP sampling, smart approvals, …) resolved its own provider
and could elect a public endpoint from whatever credential happened to be in the
environment. These tests pin the invariants that stop that:

* only the sanctioned Elite gateway and configured local routes are reachable;
* a credential's presence never changes the routing decision toward direct egress;
* every client-construction path is gated, not just the OpenAI-wire one;
* enforcement cannot be disabled from the environment;
* failures fail closed.

The autouse `_elite_aux_egress_offline` fixture in tests/conftest.py disables
enforcement for the rest of the suite; these tests re-enable it explicitly.
"""

import os

import pytest

from agent import aux_egress_policy as egress


GATEWAY = "http://100.100.175.79:8082/v1"
LITELLM = "http://100.100.175.79:4000/v1"


@pytest.fixture(autouse=True)
def _enforce(monkeypatch):
    """Undo the suite-wide neutralizer — these tests exercise enforcement."""
    monkeypatch.setattr(egress, "_ENFORCE", True)


@pytest.fixture
def _routes(monkeypatch):
    """Pin the approved-route set so tests do not depend on the user's config."""
    monkeypatch.setattr(egress, "approved_local_routes",
                        lambda: [("100.100.175.79", "4000"), ("127.0.0.1", "11434")])


# --------------------------------------------------------------- classify


@pytest.mark.parametrize("url,expected", [
    (GATEWAY, egress.GATEWAY),
    ("http://100.100.175.79:8082", egress.GATEWAY),
    (LITELLM, egress.LOCAL),
    ("http://127.0.0.1:11434/v1", egress.LOCAL),
    ("https://api.openai.com/v1", egress.FORBIDDEN),
    ("https://api.anthropic.com", egress.FORBIDDEN),
    ("https://api.deepseek.com/v1", egress.FORBIDDEN),
    ("https://openrouter.ai/api/v1", egress.FORBIDDEN),
    ("https://generativelanguage.googleapis.com/v1beta", egress.FORBIDDEN),
    ("https://bedrock-runtime.us-east-1.amazonaws.com", egress.FORBIDDEN),
    ("https://evil.example.com/v1", egress.FORBIDDEN),
])
def test_classification(_routes, url, expected):
    assert egress.classify(url)[0] == expected


@pytest.mark.parametrize("url", [
    "",
    "not a url",
    "http://",
    None,
])
def test_unclassifiable_targets_are_forbidden(_routes, url):
    assert egress.classify(url)[0] == egress.FORBIDDEN


@pytest.mark.parametrize("url", [
    # The gateway's own host on a port that is not the gateway's.
    "http://100.100.175.79:9999/v1",
    # A local-looking PORT on a remote host.
    "https://gw.remote.example:4000/v1",
    # Unapproved private / tailnet / loopback targets.
    "http://192.168.1.50:8000/v1",
    "http://100.74.115.94:9999/v1",
    "http://127.0.0.1:9999/v1",
    # Cloud metadata.
    "http://169.254.169.254/latest/meta-data/",
])
def test_near_miss_targets_are_forbidden(_routes, url):
    assert egress.classify(url)[0] == egress.FORBIDDEN


@pytest.mark.parametrize("url", [
    "http://100.100.175.79:8082@api.openai.com/v1",   # credentials-in-URL
    "http://api.openai.com@100.100.175.79:8082/v1",   # userinfo confusion
    "http://user:pass@api.openai.com/v1",
    "http://api.openai.com./v1",                       # trailing dot
    "HTTPS://API.OPENAI.COM/v1",                       # case
    # NOTE: ":08082" is deliberately NOT in this list. _host_port parses with
    # urlsplit — the same semantics httpx uses — so a zero-padded port resolves
    # to 8082, which IS the gateway and is where the client actually connects.
    # Refusing it would be an over-refusal, not a control. What matters is that
    # policy and the HTTP client never DISAGREE; see the differential test below.
    "http://evil.example.com/v1#100.100.175.79:8082",  # fragment smuggling
    "http://evil.example.com/v1?u=100.100.175.79:8082",  # query smuggling
    "http://100.100.175.79.evil.com:8082/v1",          # suffix confusion
])
def test_url_evasion_is_refused(_routes, url):
    """A shape the classifier cannot read unambiguously must not be approved."""
    assert egress.classify(url)[0] == egress.FORBIDDEN


def test_classifier_never_disagrees_with_the_http_client(_routes):
    """If we say gateway/local, the SDK must connect to the same host."""
    httpx = pytest.importorskip("httpx")
    for url in ["http://100.100.175.79:8082@api.openai.com/v1",
                "http://api.openai.com@100.100.175.79:8082/v1",
                "http://100.100.175.79:08082/v1",     # padded port
                "http://100.100.175.79:04000/v1",
                "http://100.100.175.79:9999/v1",
                GATEWAY, LITELLM]:
        verdict, _ = egress.classify(url)
        if verdict in (egress.GATEWAY, egress.LOCAL):
            hx = httpx.URL(url)
            host, port = egress._host_port(url)
            assert hx.host == host
            # Port too: the ":08082" reconciliation rests on policy and the
            # transport agreeing about the PORT, not just the host.
            assert (str(hx.port) if hx.port is not None else "") == port


def test_config_route_without_a_port_does_not_approve_every_port(monkeypatch):
    monkeypatch.setattr(egress, "approved_local_routes", lambda: [("10.0.0.9", "")])
    assert egress.classify("http://10.0.0.9:443/v1")[0] == egress.FORBIDDEN


def test_public_provider_in_config_is_not_promoted_to_local(monkeypatch):
    """Appearing under `providers:` must not make a public host 'local'."""
    monkeypatch.setattr(egress, "_config", lambda: {
        "providers": {"openai": {"base_url": "https://api.openai.com/v1"},
                      "litellm": {"base_url": LITELLM}}})
    routes = egress.approved_local_routes()
    assert ("api.openai.com", "") not in routes
    assert not any(h == "api.openai.com" for h, _ in routes)
    assert ("100.100.175.79", "4000") in routes


# --------------------------------------------------------------- enforcement


def test_assert_allowed_raises_for_forbidden(_routes):
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed("https://api.openai.com/v1", "unit-test")


def test_assert_allowed_permits_approved(_routes):
    assert egress.assert_allowed(GATEWAY, "unit-test") == egress.GATEWAY
    assert egress.assert_allowed(LITELLM, "unit-test") == egress.LOCAL


def test_blocked_exception_is_not_an_auth_error():
    """401/403 would send a local policy refusal into credential rotation."""
    exc = egress.AuxEgressBlocked("api.openai.com", "site", "direct public provider")
    assert exc.status_code == 400
    assert exc.status_code not in (401, 403)


def test_enforcement_is_not_disableable_from_the_environment(monkeypatch, tmp_path):
    """A kill switch reachable from ambient process state is not a kill switch."""
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  elite_egress_lockdown: false\n")
    for var in ("HERMES_HOME", "HERMES_CONFIG", "HERMES_CONFIG_PATH",
                "ELITE_EGRESS_LOCKDOWN", "AUX_EGRESS_LOCKDOWN",
                "HERMES_DISABLE_EGRESS_POLICY"):
        monkeypatch.setenv(var, str(tmp_path))
    assert egress.enabled() is True
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed("https://api.openai.com/v1", "unit-test")


# --------------------------------------------------------------- wiring


def test_every_client_constructor_is_gated():
    """A new SDK path must not be addable without passing the gate.

    This is a static scan rather than a behavioural test on purpose: the first
    cut of this lockdown gated only the OpenAI-wire factory, which left the
    Gemini native, Anthropic, Bedrock and Copilot-ACP constructors returning
    fully usable public-provider clients.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[2]
           / "agent" / "auxiliary_client.py").read_text().split("\n")
    constructors = ("GeminiNativeClient(", "AsyncGeminiNativeClient(",
                    "build_anthropic_client(", "build_anthropic_bedrock_client(",
                    "CopilotACPClient(", "CodexAuxiliaryClient(",
                    "AsyncOpenAI(", "OpenAI(")
    ungated = []
    for i, line in enumerate(src):
        stripped = line.strip()
        if not any(c in line for c in constructors) or "def " in line:
            continue
        # Skip prose: the module and class docstrings name constructors.
        if stripped.startswith(("#", chr(34) * 3, chr(39) * 3, "*", "-")):
            continue
        if "``" in stripped or "_create_openai_client" in stripped:
            continue
        # Distinguish CONSTRUCTION from WRAPPING. A call that establishes an
        # endpoint names one (api_key=/base_url=/region) and must be gated. A
        # call that merely wraps an existing client object inherits the gate
        # that produced it, and gating it again would be theatre.
        call = stripped.split("(", 1)[1] if "(" in stripped else ""
        establishes_endpoint = any(t in call for t in
                                   ("api_key", "base_url", "region", "token="))
        if not establishes_endpoint:
            continue
        if "_create_openai_client(" in "\n".join(src[max(0, i - 6):i]):
            continue
        # Scan back to the start of the enclosing function: a gate placed at the
        # top of a function (so it runs before any early return) is still a gate,
        # and a fixed-size window would miss it.
        start = 0
        for j in range(i, -1, -1):
            if src[j].startswith("def ") or src[j].startswith("    def "):
                start = j
                break
        if "_gated_client(" not in "\n".join(src[start:i]):
            ungated.append(f"{i + 1}: {line.strip()[:70]}")
    assert not ungated, "ungated client construction:\n" + "\n".join(ungated)


def test_ambient_credential_discovery_cannot_elect_a_public_provider(_routes):
    """The api-key step STAYS in the chain, but cannot yield a public provider.

    An earlier fix deleted the step outright. That was the wrong instrument: it
    is also the only way LM Studio (127.0.0.1:1234, an approved local route) is
    discovered, so deleting it starved a legitimate local backend. The step is
    retained and each candidate inside it is refused individually.
    """
    import agent.auxiliary_client as ac
    from hermes_cli.auth import PROVIDER_REGISTRY

    assert "api-key" in [name for name, _ in ac._get_provider_chain()], \
        "removing the step starves LM Studio, an approved local route"

    # No public provider in the registry may survive classification.
    for pid, pconfig in PROVIDER_REGISTRY.items():
        url = getattr(pconfig, "inference_base_url", "") or ""
        if not url:
            continue
        verdict = egress.classify(url)[0]
        if any(pub in url for pub in egress.KNOWN_PUBLIC_PROVIDER_HOSTS):
            assert verdict == egress.FORBIDDEN, f"{pid} ({url}) not refused"


def test_local_provider_seeding_is_still_permitted():
    """The credential-pool gate must not starve APPROVED routes."""
    from agent import credential_pool as cp

    for approved in ("lmstudio", "custom", "litellm"):
        assert cp._provider_seed_allowed(approved) is True, approved
    for refused in ("openai", "anthropic", "openrouter", "nous", "gemini",
                    "zai", "deepseek", "an-unregistered-provider"):
        assert cp._provider_seed_allowed(refused) is False, refused


def test_gateway_client_gets_the_gateway_credential(monkeypatch, _routes):
    import agent.auxiliary_client as ac
    monkeypatch.setattr(egress, "gateway_credential", lambda: "synthetic-gw-key")
    client = ac._create_openai_client(api_key="no-key-required", base_url=GATEWAY)
    assert client.api_key == "synthetic-gw-key"


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1",
    "https://api.anthropic.com",
    "https://openrouter.ai/api/v1",
])
def test_openai_factory_refuses_public_endpoints(_routes, url):
    import agent.auxiliary_client as ac
    with pytest.raises(egress.AuxEgressBlocked):
        ac._create_openai_client(api_key="synthetic-not-a-real-key", base_url=url)


# --------------------------------------------------------------- hygiene


def test_metrics_labels_are_low_cardinality(_routes):
    egress.COUNTERS.clear()
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed("https://api.openai.com/v1", "unit-test")
    lines = egress.metrics_text()
    assert lines
    for line in lines:
        for banned in ("api_key", "token", "prompt", "content", "user="):
            assert banned not in line


# --------------------------------------------------------------- async path


@pytest.mark.parametrize("url,allowed", [
    (GATEWAY, True),
    (LITELLM, True),
    ("http://127.0.0.1:11434/v1", True),
    ("https://api.openai.com/v1", False),
    ("https://generativelanguage.googleapis.com/v1beta", False),
])
def test_async_client_factory_is_gated_and_still_works(_routes, url, allowed):
    """Behavioural, not static.

    The first attempt at gating this function referenced a name assigned
    further down, making it a function-local — so the gate raised
    UnboundLocalError on EVERY async call, approved targets included, and the
    whole async auxiliary stack (vision, video, the plugin bridge) was dead.
    The static "is there a gate in this function" scan passed happily. Only
    calling it catches that, so this test calls it.
    """
    import types
    import agent.auxiliary_client as ac

    sync = types.SimpleNamespace(base_url=url, api_key="synthetic-not-a-real-key")
    if allowed:
        client, model = ac._to_async_client(sync, "test-model")
        assert client is not None
        assert model == "test-model"
    else:
        with pytest.raises(egress.AuxEgressBlocked):
            ac._to_async_client(sync, "test-model")


def test_a_refused_backend_does_not_abort_the_resolution_chain(_routes, monkeypatch):
    """One forbidden entry must not deny an approved entry further down."""
    import agent.auxiliary_client as ac

    def _boom(*a, **k):
        raise egress.AuxEgressBlocked("openrouter.ai", "test", "direct public provider")

    monkeypatch.setattr(ac, "_try_openrouter", _boom, raising=False)
    try:
        ac._resolve_auto(main_runtime={"provider": "", "model": "",
                                       "base_url": "", "api_key": ""},
                         task="compression")
    except egress.AuxEgressBlocked:
        raise AssertionError("a refused chain entry aborted the whole resolution")


def test_userinfo_cannot_smuggle_an_approved_host(monkeypatch):
    """The CRITICAL finding: a config route must not be able to approve itself.

    `_host_port` used to split on the first ":" and never strip RFC-3986
    userinfo, so `http://127.0.0.1:11434@evil.example.com/v1` read as host
    127.0.0.1 while httpx resolves evil.example.com. Because the allowlist
    builder and the classifier share the parser, such a URL in config
    registered a route that then MATCHED ITSELF and classified LOCAL — a full
    bypass from an input this module treats as untrusted.
    """
    httpx = pytest.importorskip("httpx")
    hostile = "http://127.0.0.1:11434@evil.example.com/v1"

    # The parser must agree with the transport, or fail closed.
    host, _ = egress._host_port(hostile)
    assert host in ("", httpx.URL(hostile).host)

    # And a config entry naming it must not become an approved route.
    monkeypatch.setattr(egress, "_config",
                        lambda: {"providers": {"evil": {"base_url": hostile}}})
    assert egress.classify(hostile)[0] == egress.FORBIDDEN
    assert not any(h == "127.0.0.1" and "@" in p
                   for h, p in egress.approved_local_routes())


def test_link_local_metadata_service_is_never_local(monkeypatch):
    """169.254.169.254 is the cloud metadata service, not a local model."""
    monkeypatch.setattr(egress, "_config", lambda: {
        "providers": {"meta": {"base_url": "http://169.254.169.254/latest"}}})
    assert egress._is_plausibly_local("169.254.169.254") is False
    assert egress.classify("http://169.254.169.254/latest")[0] == egress.FORBIDDEN


def test_every_set_proxy_variable_is_classified(monkeypatch, _routes):
    """An approved decoy must not hide an attacker value in another variable."""
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:11434")   # approved decoy
    monkeypatch.setenv("HTTP_PROXY", "http://evil.example.com:3128")
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed(GATEWAY, "unit-test")


def test_probe_gemini_tier_reports_rather_than_crashing(_routes):
    """The gate must not turn a best-effort probe into an unhandled traceback.

    `probe_gemini_tier`'s documented contract is that "unknown" means the probe
    failed and callers should proceed. Its only production caller
    (hermes_cli/model_setup_flows.py) has no handler, so raising from the gate
    broke `hermes setup` / `hermes model` outright.
    """
    from agent import gemini_native_adapter as gna

    assert gna.probe_gemini_tier(
        "synthetic-not-a-real-key",
        base_url="https://generativelanguage.googleapis.com/v1beta") == "unknown"


def test_trajectory_compressor_gates_both_clients(_routes, monkeypatch):
    """Its shipped default is OpenRouter + OPENROUTER_API_KEY."""
    import trajectory_compressor as tc

    src = open(tc.__file__).read()
    # Both the sync and async constructions must be gated.
    assert src.count("assert_allowed") >= 2, "sync and async clients must both gate"
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed("https://openrouter.ai/api/v1",
                              "trajectory_compressor.custom_endpoint")


@pytest.mark.parametrize("var", [
    "Http_Proxy", "HTTP_Proxy", "http_PROXY", "hTTP_PROXY",
    "All_Proxy", "ALL_proxy", "Https_Proxy",
])
def test_mixed_case_proxy_variables_are_seen(monkeypatch, _routes, var):
    """POSIX env vars are case-sensitive; the client's resolver lower-cases them.

    A fixed list of exact names (HTTP_PROXY, ...) therefore missed `Http_Proxy`,
    which every trust_env client honours — a credential-carrying request went to
    an attacker proxy while the policy reported no proxy at all.
    """
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(var, "http://evil-proxy.example.com:3128")
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed(GATEWAY, "unit-test")


@pytest.mark.parametrize("host", [
    "2002:a9fe:a9fe::",   # 6to4 embedding the metadata service
    "2002:0808:0808::",   # 6to4 embedding a public IPv4
    "2001:0:1::",         # teredo
    "0.0.0.0", "::", "255.255.255.255",
])
def test_embedded_ipv4_and_special_addresses_are_not_local(host):
    """ipaddress reports 6to4/teredo as private, but they embed arbitrary IPv4."""
    assert egress._is_plausibly_local(host) is False


def test_platform_system_proxy_is_classified(monkeypatch, _routes):
    """Clients resolve via getproxies(), which falls back to the PLATFORM proxy.

    Reading only getproxies_environment() left the macOS sysconf / Windows
    registry channel invisible to the policy while every trust_env client
    honoured it — and the Windows registry key is user-writable with no
    privilege required.
    """
    import urllib.request

    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(urllib.request, "getproxies",
                        lambda: {"http": "http://evil-proxy.example.com:3128"})
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed("http://127.0.0.1:11434/v1", "unit-test")


# --------------------------------------------------- proxy vs NO_PROXY vs relay
#
# Two defects lived here, both reproduced against the shipped policy before this
# section existed:
#
# 1. NO_PROXY was classified as a proxy. urllib's getproxies() turns every
#    `*_proxy` variable into a key, so `NO_PROXY=localhost,127.0.0.1` arrived as
#    key "no" with that value — and the policy read the value as a proxy URL,
#    classified it FORBIDDEN and refused ordinary gateway traffic. A bypass list
#    is the statement that a host is reached DIRECTLY; it is the opposite of a
#    proxy.
#
# 2. An approved inference endpoint was accepted as a forward proxy.
#    `HTTP_PROXY=http://127.0.0.1:11434` passed because Ollama is an approved
#    local model — so the gateway credential and the whole prompt body would be
#    relayed in cleartext to whatever listens on that port. Receiving inference
#    and relaying it are different trust decisions.

_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
               "http_proxy", "https_proxy", "all_proxy", "no_proxy")

OLLAMA = "http://127.0.0.1:11434/v1"


@pytest.fixture
def _clean_proxy_env(monkeypatch):
    """No inherited proxy state, and no platform proxy, unless a test sets one."""
    import urllib.request
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(urllib.request, "getproxies",
                        lambda: dict(_env_proxies()))
    return monkeypatch


def _env_proxies():
    """getproxies_environment() semantics: lower-cased name, `_proxy` stripped."""
    out = {}
    for name, value in os.environ.items():
        low = name.lower()
        if low.endswith("_proxy") and value:
            out[low[:-6]] = value
    return out


@pytest.fixture
def _no_sanctioned_proxy(monkeypatch):
    """Default posture: no forward proxy is approved, so any proxy fails closed."""
    monkeypatch.setattr(egress, "approved_proxies", lambda: [])


# -- 1/2/3: NO_PROXY is bypass metadata, never a proxy destination -------------


def test_no_proxy_value_is_not_classified_as_a_proxy(
        _routes, _clean_proxy_env, _no_sanctioned_proxy):
    """REGRESSION: `NO_PROXY=localhost,127.0.0.1` blocked all gateway traffic.

    The gateway is not covered by that bypass list and no proxy is configured,
    so the correct answer is a direct, allowed request.
    """
    _clean_proxy_env.setenv("NO_PROXY", "localhost,127.0.0.1")
    assert egress.candidate_proxies(GATEWAY) == []
    assert egress.assert_allowed(GATEWAY, "unit-test") == egress.GATEWAY


def test_no_proxy_covering_the_destination_matches_the_transport(
        _routes, _clean_proxy_env, _no_sanctioned_proxy):
    """A real proxy is set, but NO_PROXY exempts the gateway — as httpx agrees."""
    _clean_proxy_env.setenv("HTTP_PROXY", "http://evil.example.com:3128")
    _clean_proxy_env.setenv("NO_PROXY", egress.SANCTIONED_GATEWAY_HOST)
    assert egress.no_proxy_bypasses(egress.SANCTIONED_GATEWAY_HOST) is True
    assert egress.candidate_proxies(GATEWAY) == []
    assert egress.assert_allowed(GATEWAY, "unit-test") == egress.GATEWAY
    # ... and an unexempted destination through the same variable still blocks.
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed(OLLAMA, "unit-test")


def test_lowercase_no_proxy_is_honoured(
        _routes, _clean_proxy_env, _no_sanctioned_proxy):
    """POSIX keeps `no_proxy` distinct from `NO_PROXY`; clients honour both."""
    _clean_proxy_env.setenv("http_proxy", "http://evil.example.com:3128")
    _clean_proxy_env.setenv("no_proxy", egress.SANCTIONED_GATEWAY_HOST)
    assert egress.candidate_proxies(GATEWAY) == []
    assert egress.assert_allowed(GATEWAY, "unit-test") == egress.GATEWAY


@pytest.mark.parametrize("no_proxy,host,expected", [
    ("*", "evil.example.com", True),
    ("localhost,127.0.0.1", "100.100.175.79", False),
    ("localhost", "localhost", True),
    ("127.0.0.1", "127.0.0.1", True),
    ("127.0.0.1", "127.0.0.2", False),
    ("google.com", "www.google.com", True),
    ("google.com", "wwwgoogle.com", False),
    (".google.com", "www.google.com", True),
    ("192.168.0.0/16", "192.168.4.7", True),
    ("192.168.0.0/16", "10.0.0.1", False),
    ("100.100.175.79:8082", "100.100.175.79", True),
    ("", "100.100.175.79", False),
])
def test_no_proxy_matching_rules(no_proxy, host, expected):
    """Follows curl's CURLOPT_NOPROXY rules, which httpx mirrors."""
    assert egress.no_proxy_bypasses(host, no_proxy) is expected


# -- 4/5/6/7: every channel a transport can actually use is still seen ---------


@pytest.mark.parametrize("var", ["HTTP_PROXY", "Http_Proxy", "hTTP_PROXY"])
def test_mixed_case_http_proxy_is_still_detected(
        _routes, _clean_proxy_env, _no_sanctioned_proxy, var):
    """getproxies_environment() lower-cases NAMES, so mixed case still resolves."""
    _clean_proxy_env.setenv(var, "http://evil.example.com:3128")
    assert egress.candidate_proxies(GATEWAY) == ["http://evil.example.com:3128"]
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed(GATEWAY, "unit-test")


@pytest.mark.parametrize("var", ["HTTPS_PROXY", "ALL_PROXY"])
def test_https_and_all_proxy_are_still_detected(
        _routes, _clean_proxy_env, _no_sanctioned_proxy, var):
    """A destination-scheme mismatch is not a reason to ignore a set proxy.

    Auxiliary clients are built for several destinations and several SDKs; an
    approved-looking decoy in one variable must not hide an attacker value in
    another.
    """
    _clean_proxy_env.setenv(var, "http://evil.example.com:3128")
    assert "http://evil.example.com:3128" in egress.candidate_proxies(GATEWAY)
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed(GATEWAY, "unit-test")


def test_platform_proxy_is_still_detected(
        _routes, _clean_proxy_env, _no_sanctioned_proxy):
    """getproxies() falls back to macOS sysconf / the Windows registry."""
    import urllib.request
    _clean_proxy_env.setattr(
        urllib.request, "getproxies",
        lambda: {"http": "http://evil-proxy.example.com:3128"})
    assert egress.candidate_proxies(OLLAMA) == ["http://evil-proxy.example.com:3128"]
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed(OLLAMA, "unit-test")


def test_unrelated_proxy_variables_are_not_transport_proxies(
        _routes, _clean_proxy_env, _no_sanctioned_proxy):
    """`TRAVIS_APT_PROXY` is not something an HTTP client can route through.

    httpx mounts only the http/https/all keys and drops the rest by name; a
    policy that classified every `*_PROXY` variable both blocked traffic no
    transport would have proxied and misdescribed what the transport does.
    """
    _clean_proxy_env.setenv("TRAVIS_APT_PROXY", "http://apt-cache.example.com:3142")
    _clean_proxy_env.setenv("FTP_PROXY", "http://ftp-cache.example.com:2121")
    assert egress.candidate_proxies(GATEWAY) == []
    assert egress.assert_allowed(GATEWAY, "unit-test") == egress.GATEWAY


# -- 8/9/10: a proxy is a relay, not a destination -----------------------------


def test_public_proxy_is_blocked(_routes, _clean_proxy_env, _no_sanctioned_proxy):
    _clean_proxy_env.setenv("HTTP_PROXY", "http://evil.example.com:3128")
    with pytest.raises(egress.AuxEgressBlocked) as exc:
        egress.assert_allowed(GATEWAY, "unit-test")
    assert "proxied" in str(exc.value)


@pytest.mark.parametrize("proxy,why", [
    ("http://127.0.0.1:11434", "Ollama is an approved INFERENCE route"),
    ("http://100.100.175.79:4000", "LiteLLM is an approved INFERENCE route"),
    ("http://100.100.175.79:8082", "the gateway itself is a destination, not a relay"),
])
def test_approved_inference_endpoint_is_not_an_approved_forward_proxy(
        _routes, _clean_proxy_env, _no_sanctioned_proxy, proxy, why):
    """REGRESSION: proxy_is_approved() accepted any LOCAL/GATEWAY classification.

    A forward proxy terminates the connection: it receives ELITE_GATEWAY_KEY and
    the entire request body for everything it relays. Being trusted to answer an
    inference request addressed to it does not extend to that. {why}
    """
    _clean_proxy_env.setenv("HTTP_PROXY", proxy)
    assert egress.classify(proxy)[0] in (egress.LOCAL, egress.GATEWAY)   # destination-approved
    ok, reason = egress.proxy_is_approved(GATEWAY)
    assert ok is False, why
    assert "proxy" in reason
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed(GATEWAY, "unit-test")


def test_approved_proxies_is_not_the_local_route_list(monkeypatch):
    """The two allowlists must not be wired to the same source."""
    monkeypatch.setattr(egress, "approved_local_routes",
                        lambda: [("127.0.0.1", "11434"), ("100.100.175.79", "4000")])
    monkeypatch.setattr(egress, "_config", lambda: {})
    assert egress.approved_proxies() == []


@pytest.mark.parametrize("configured,expected", [
    (["http://127.0.0.1:3128"], [("127.0.0.1", "3128")]),
    ("http://127.0.0.1:3128", [("127.0.0.1", "3128")]),
    (["http://evil.example.com:3128"], []),      # not plausibly local
    (["https://api.openai.com:443"], []),        # public provider
    (["http://127.0.0.1"], []),                  # no explicit port
    ([], []),
])
def test_approved_proxies_is_explicit_and_bounded(monkeypatch, configured, expected):
    """Config can name a sanctioned proxy, but not an arbitrary off-box one.

    This config path resolves through HERMES_HOME and is writable by Hermes' own
    file tools, so an unbounded proxy allowlist here would be a one-line
    exfiltration switch.
    """
    monkeypatch.setattr(egress, "_config", lambda: {
        "auxiliary": {"approved_egress_proxies": configured}})
    assert egress.approved_proxies() == expected


def test_explicitly_sanctioned_proxy_is_allowed(_routes, _clean_proxy_env):
    """Fail-closed is the default, not a refusal to support a real proxy."""
    _clean_proxy_env.setattr(egress, "_config", lambda: {
        "auxiliary": {"approved_egress_proxies": ["http://127.0.0.1:3128"]}})
    _clean_proxy_env.setenv("HTTP_PROXY", "http://127.0.0.1:3128")
    assert egress.assert_allowed(GATEWAY, "unit-test") == egress.GATEWAY
    # A different port on the same sanctioned host is a different service.
    _clean_proxy_env.setenv("HTTP_PROXY", "http://127.0.0.1:3129")
    with pytest.raises(egress.AuxEgressBlocked):
        egress.assert_allowed(GATEWAY, "unit-test")


# -- 11/12/13: the direct, no-proxy paths still work ---------------------------


@pytest.mark.parametrize("url,expected", [
    (GATEWAY, egress.GATEWAY),
    (LITELLM, egress.LOCAL),
    (OLLAMA, egress.LOCAL),
])
def test_direct_routes_pass_with_no_proxy_configured(
        _routes, _clean_proxy_env, _no_sanctioned_proxy, url, expected):
    assert egress.effective_proxy(url) is None
    assert egress.assert_allowed(url, "unit-test") == expected


# -- transport parity ----------------------------------------------------------


@pytest.mark.parametrize("env", [
    {},
    {"NO_PROXY": "localhost,127.0.0.1"},
    {"NO_PROXY": "100.100.175.79"},
    {"no_proxy": "100.100.175.79"},
    {"HTTP_PROXY": "http://evil.example.com:3128"},
    {"HTTP_PROXY": "http://evil.example.com:3128", "NO_PROXY": "100.100.175.79"},
    {"HTTP_PROXY": "http://evil.example.com:3128", "NO_PROXY": "*"},
    {"ALL_PROXY": "http://evil.example.com:3128"},
    {"TRAVIS_APT_PROXY": "http://apt-cache.example.com:3142"},
])
@pytest.mark.parametrize("url", [GATEWAY, LITELLM, OLLAMA])
def test_policy_and_real_transport_agree_on_whether_a_proxy_is_used(
        _routes, _clean_proxy_env, _no_sanctioned_proxy, env, url):
    """The policy must never report "direct" for a request httpx would proxy.

    Parity is asserted in the direction that matters: whenever the real client
    would route through a proxy, the policy must have seen one. The policy may
    additionally flag a proxy httpx would not pick for this particular scheme —
    other auxiliary SDKs and destinations do, and an approved-looking decoy in
    one variable must not mask an attacker value in another.
    """
    import httpx

    for name, value in env.items():
        _clean_proxy_env.setenv(name, value)

    client = httpx.Client(trust_env=True)
    try:
        transport_proxies = client._transport_for_url(httpx.URL(url)) is not client._transport
    finally:
        client.close()

    policy_proxies = bool(egress.candidate_proxies(url))
    if transport_proxies:
        assert policy_proxies, f"transport proxies {url} but policy saw none: {env}"
    if not policy_proxies:
        assert not transport_proxies
        assert egress.proxy_is_approved(url)[0] is True


# ------------------------------------------- probe_gemini_tier must fail CLOSED
#
# `probe_gemini_tier` POSTs to :generateContent through an httpx client it builds
# itself, so it never passes `auxiliary_client._gated_client` and carries its own
# gate. That gate had a fail-OPEN branch:
#
#     except ImportError:
#         pass                      # <- fell through into the HTTP POST
#
# The one state in which the lockdown cannot express an opinion — its module
# missing, unimportable or half-initialised — was the one state in which the key
# and prompt went straight to generativelanguage.googleapis.com. The invariant is
# the opposite: policy unavailable, broken, missing or refusing all mean zero
# external provider calls.
#
# These tests instrument the REAL egress boundary (httpx's transport handler and
# socket.connect), not probe_gemini_tier's return value, so a test cannot pass
# merely because a mock returned "unknown".

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
SYNTHETIC_KEY = "synthetic-not-a-real-key-0000"


class _EgressCounter:
    """Counts every attempt to leave the process, at the lowest layers available."""

    def __init__(self) -> None:
        self.transport_calls = []
        self.socket_calls = []

    @property
    def total(self) -> int:
        return len(self.transport_calls) + len(self.socket_calls)


@pytest.fixture
def _http_boundary(monkeypatch):
    """Make any real egress attempt observable and non-networked.

    Patched at httpx's HTTPTransport.handle_request (the actual request/response
    boundary of the client probe_gemini_tier constructs) AND at socket.connect,
    so a call that somehow bypasses httpx is still counted rather than silently
    reaching the network.
    """
    import socket

    import httpx

    counter = _EgressCounter()

    def _fake_handle_request(self, request):
        counter.transport_calls.append(str(request.url))
        raise httpx.ConnectError("blocked by test boundary", request=request)

    def _fake_connect(self, address):
        counter.socket_calls.append(address)
        raise AssertionError(f"test attempted a real socket connection to {address}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _fake_handle_request)
    monkeypatch.setattr(socket.socket, "connect", _fake_connect)
    return counter


@pytest.fixture
def _policy_unimportable(monkeypatch):
    """`from agent import aux_egress_policy` raises ImportError.

    Both halves are needed: `agent.aux_egress_policy` is already an attribute of
    the imported `agent` package, so poisoning sys.modules alone would still
    resolve via getattr and never reach the import machinery.
    """
    import sys

    import agent as _agent_pkg

    monkeypatch.delattr(_agent_pkg, "aux_egress_policy", raising=False)
    monkeypatch.setitem(sys.modules, "agent.aux_egress_policy", None)


@pytest.fixture
def _policy_missing(monkeypatch):
    """The module does not exist at all: ModuleNotFoundError, not ImportError."""
    import importlib.abc
    import importlib.machinery
    import sys

    import agent as _agent_pkg

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "agent.aux_egress_policy":
                raise ModuleNotFoundError(
                    "No module named 'agent.aux_egress_policy'",
                    name="agent.aux_egress_policy")
            return None

    monkeypatch.delattr(_agent_pkg, "aux_egress_policy", raising=False)
    monkeypatch.delitem(sys.modules, "agent.aux_egress_policy", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Blocker()] + list(sys.meta_path))


def _install_policy_stub(monkeypatch, stub):
    """Put an arbitrary object in the policy module's place."""
    import sys

    import agent as _agent_pkg

    monkeypatch.setattr(_agent_pkg, "aux_egress_policy", stub, raising=False)
    monkeypatch.setitem(sys.modules, "agent.aux_egress_policy", stub)


# -- 1: an explicit refusal never reaches the network --------------------------


def test_policy_refusal_makes_zero_gemini_http_calls(_http_boundary, monkeypatch):
    """The real policy, enforcing, refuses the public Gemini endpoint."""
    from agent import gemini_native_adapter as gna

    monkeypatch.setattr(egress, "_ENFORCE", True)
    monkeypatch.setattr(egress, "approved_local_routes",
                        lambda: [("127.0.0.1", "11434")])
    assert egress.classify(GEMINI_URL)[0] == egress.FORBIDDEN

    assert gna.probe_gemini_tier(SYNTHETIC_KEY, base_url=GEMINI_URL) == "unknown"
    assert _http_boundary.total == 0, _http_boundary.transport_calls


# -- 2: the policy module cannot be imported -----------------------------------


def test_policy_unimportable_makes_zero_gemini_http_calls(
        _http_boundary, _policy_unimportable):
    """REGRESSION: `except ImportError: pass` fell through into the POST."""
    from agent import gemini_native_adapter as gna

    with pytest.raises(ImportError):
        from agent import aux_egress_policy  # noqa: F401

    assert gna.probe_gemini_tier(SYNTHETIC_KEY, base_url=GEMINI_URL) == "unknown"
    assert _http_boundary.total == 0, _http_boundary.transport_calls


def test_policy_module_missing_makes_zero_gemini_http_calls(
        _http_boundary, _policy_missing):
    """The module is absent from disk, not merely poisoned in sys.modules.

    Note what the import actually raises: `_handle_fromlist` swallows the
    ModuleNotFoundError and the `from agent import ...` form surfaces a plain
    `ImportError: cannot import name`. So "the module was deleted" landed in the
    very same `except ImportError: pass` branch as a transient import glitch —
    both fell through into the POST.
    """
    from agent import gemini_native_adapter as gna

    with pytest.raises(ImportError):
        from agent import aux_egress_policy  # noqa: F401

    assert gna.probe_gemini_tier(SYNTHETIC_KEY, base_url=GEMINI_URL) == "unknown"
    assert _http_boundary.total == 0, _http_boundary.transport_calls


# -- 3/4: a broken or half-initialised policy is not a licence to proceed ------


def test_policy_evaluation_exception_makes_zero_gemini_http_calls(
        _http_boundary, monkeypatch):
    """An unexpected error inside the check is not an approval."""
    from agent import gemini_native_adapter as gna

    class _Exploding:
        def assert_allowed(self, *a, **k):
            raise RuntimeError("policy backend exploded")

    _install_policy_stub(monkeypatch, _Exploding())

    assert gna.probe_gemini_tier(SYNTHETIC_KEY, base_url=GEMINI_URL) == "unknown"
    assert _http_boundary.total == 0, _http_boundary.transport_calls


def test_broken_partial_policy_returns_unknown_without_crashing(
        _http_boundary, monkeypatch):
    """A module present but missing assert_allowed must not raise to the caller.

    `probe_gemini_tier`'s sole caller (hermes_cli/model_setup_flows.py) has no
    handler, so raising here breaks `hermes setup` / `hermes model` outright.
    "unknown" is a report of failure, not a fallback.
    """
    from agent import gemini_native_adapter as gna

    class _Partial:
        pass                              # no assert_allowed at all

    _install_policy_stub(monkeypatch, _Partial())

    assert gna.probe_gemini_tier(SYNTHETIC_KEY, base_url=GEMINI_URL) == "unknown"
    assert _http_boundary.total == 0, _http_boundary.transport_calls


@pytest.mark.parametrize("boom", [
    BaseException("not an Exception subclass"),
    MemoryError("out of memory"),
])
def test_policy_failure_never_yields_a_probe(_http_boundary, monkeypatch, boom):
    """Even failures that propagate past the handler must not send a request."""
    from agent import gemini_native_adapter as gna

    class _Exploding:
        def assert_allowed(self, *a, **k):
            raise boom

    _install_policy_stub(monkeypatch, _Exploding())

    try:
        gna.probe_gemini_tier(SYNTHETIC_KEY, base_url=GEMINI_URL)
    except BaseException:
        pass
    assert _http_boundary.total == 0, _http_boundary.transport_calls


# -- 5: an approved route still behaves exactly as before ----------------------


def test_approved_route_still_probes(_http_boundary, monkeypatch):
    """Fail-closed must not mean fail-always: a sanctioned route still runs."""
    from agent import gemini_native_adapter as gna

    monkeypatch.setattr(egress, "_ENFORCE", True)
    monkeypatch.setattr(egress, "approved_local_routes",
                        lambda: [("127.0.0.1", "11434")])
    local = "http://127.0.0.1:11434/v1beta"
    assert egress.classify(local)[0] == egress.LOCAL

    gna.probe_gemini_tier(SYNTHETIC_KEY, base_url=local)
    assert len(_http_boundary.transport_calls) == 1
    assert _http_boundary.transport_calls[0].startswith("http://127.0.0.1:11434/")


@pytest.mark.parametrize("headers,status,expected", [
    ({"x-ratelimit-limit-requests-per-day": "250"}, 200, "free"),
    ({"x-ratelimit-limit-requests-per-day": "1500"}, 200, "paid"),
    ({}, 200, "paid"),
    ({}, 429, "paid"),
    ({}, 500, "unknown"),
])
def test_approved_route_tier_parsing_is_unchanged(
        monkeypatch, headers, status, expected):
    """The gate changed; what the probe does once allowed did not."""
    import httpx

    from agent import gemini_native_adapter as gna

    monkeypatch.setattr(egress, "_ENFORCE", True)
    monkeypatch.setattr(egress, "approved_local_routes",
                        lambda: [("127.0.0.1", "11434")])
    monkeypatch.setattr(
        httpx.HTTPTransport, "handle_request",
        lambda self, request: httpx.Response(status, headers=headers, json={}))

    assert gna.probe_gemini_tier(
        SYNTHETIC_KEY, base_url="http://127.0.0.1:11434/v1beta") == expected


# -- 6/7: no direct call and no ambient fallback from the failure handling -----


@pytest.mark.parametrize("mode", ["refused", "unimportable", "exploding"])
def test_no_direct_or_ambient_fallback_after_a_policy_failure(
        _http_boundary, monkeypatch, mode):
    """"unknown" must be the end of the road, not a hand-off to another provider.

    Anything that resolves an ambient credential or builds a second client would
    be the auxiliary bypass this lockdown exists to remove, so those entry points
    are wired to fail the test if the failure path touches them.
    """
    import sys

    from agent import gemini_native_adapter as gna

    touched = []

    import agent.auxiliary_client as aux
    for name in ("_create_openai_client", "_gated_client"):
        if hasattr(aux, name):
            monkeypatch.setattr(aux, name, lambda *a, **k: touched.append(name))
    import agent.credential_pool as pool
    for name in ("load_pool", "get_credential"):
        if hasattr(pool, name):
            monkeypatch.setattr(pool, name, lambda *a, **k: touched.append(name))

    if mode == "refused":
        monkeypatch.setattr(egress, "_ENFORCE", True)
        monkeypatch.setattr(egress, "approved_local_routes", lambda: [])
    elif mode == "unimportable":
        import agent as _agent_pkg
        monkeypatch.delattr(_agent_pkg, "aux_egress_policy", raising=False)
        monkeypatch.setitem(sys.modules, "agent.aux_egress_policy", None)
    else:
        class _Exploding:
            def assert_allowed(self, *a, **k):
                raise RuntimeError("policy backend exploded")
        _install_policy_stub(monkeypatch, _Exploding())

    # A key IS present: the bug class this guards against is "no approval, so
    # fall back to whatever credential is lying around".
    monkeypatch.setenv("GEMINI_API_KEY", SYNTHETIC_KEY)
    monkeypatch.setenv("GOOGLE_API_KEY", SYNTHETIC_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", SYNTHETIC_KEY)

    assert gna.probe_gemini_tier(SYNTHETIC_KEY, base_url=GEMINI_URL) == "unknown"
    assert _http_boundary.total == 0, _http_boundary.transport_calls
    assert touched == [], f"failure path reached auxiliary fallback: {touched}"


def test_probe_gate_has_no_fall_through_branch():
    """Source-level canary: no handler around the gate may resume the probe.

    The defect was a second `except` clause whose body was `pass`. Asserting on
    behaviour alone would let an equivalent branch be reintroduced for some other
    exception type without any test noticing.
    """
    import inspect

    from agent import gemini_native_adapter as gna

    src = inspect.getsource(gna.probe_gemini_tier)
    gate = src.split("key = (api_key")[0]
    assert "assert_allowed" in gate
    statements = [ln.strip() for ln in gate.splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    handlers = [ln for ln in statements if ln.startswith("except")]
    assert handlers, "the gate must be guarded"
    for handler in handlers:
        assert handler == "except Exception:", f"unexpected gate handler: {handler}"
    assert "pass" not in statements, \
        "a gate handler that falls through re-opens the bypass"
    assert statements.count('return "unknown"') == len(handlers)
