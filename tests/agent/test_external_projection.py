"""Tests for the Elite external-safe prompt/context projection layer.

Every protected identifier here is SYNTHETIC. The policy engine is stubbed with
a matcher of the same shape as the real root-owned guard, so these tests run
anywhere — including a machine with no protected-project policy installed.

The invariants under test are the ones that make the layer safe rather than
merely useful:

* framework contamination is removed and the request proceeds;
* user content and non-framework tool results are NEVER rewritten, only refused;
* an unattributable match refuses instead of guessing;
* a policy that cannot be trusted refuses;
* local targets are returned untouched;
* the final scan is as strong as the guard, including across whitespace that
  JSON serialization would have escaped.
"""

import re
import sys
import types

import pytest

from agent import external_projection as ep


SYNTH = "zzz-synthetic-alpha"
PREAMBLE = "## Skills (mandatory)"
GATEWAY = "http://gw.example:8082/v1"


def _stub_guard(patterns=(r"zzz-synthetic-[a-z]+",), policy_ok=True,
                baseline_ok=True, min_rules=1):
    """A stand-in with the same surface the real guard exposes."""
    mod = types.SimpleNamespace()
    compiled = [re.compile(p) for p in patterns]

    def normalize(text):
        t = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", str(text)).lower()
        t = re.sub(r"[\s_.]+", "-", t)
        return re.sub(r"-{2,}", "-", t)

    def match(text):
        if not text:
            return None
        norm = normalize(text)
        for rx in compiled:
            if rx.search(norm):
                return rx.pattern
        return None

    mod.match = match
    mod.normalize = normalize
    mod._PATTERNS = list(patterns)
    mod._POLICY_OK = policy_ok
    mod._BASELINE_OK = baseline_ok
    mod._BASELINE_REASON = "ok" if baseline_ok else "policy_hash_mismatch"
    mod.MIN_RULES = min_rules
    return mod


@pytest.fixture
def guard(monkeypatch):
    """Install a trusted stub policy engine."""
    g = _stub_guard()
    monkeypatch.setattr(ep, "_load_guard", lambda: g)
    ep._guard_cache.clear()
    return g


def _sysmsg(entries, memory=()):
    idx = "\n".join(f"    - {n}: {d}" for n, d in entries)
    sep = "═" * 46
    mem = ""
    if memory:
        mem = (f"\n\n{sep}\nMEMORY (your personal notes) [1% — 1/100 chars]\n"
               f"{sep}\n" + "\n§\n".join(memory))
    return (f"You are Hermes.\n\n{PREAMBLE}\nScan the skills below.\n"
            f"<available_skills>\n  devops:\n{idx}\n</available_skills>\n" + mem)


def _run(api_kwargs, provider="elite-gateway", base_url=GATEWAY):
    return ep.guard_outbound(api_kwargs, provider=provider, base_url=base_url,
                             request_id="test")


# --------------------------------------------------------------- framework


def test_contaminated_skill_index_entry_is_dropped_and_request_proceeds(guard):
    r = _run({"model": "m", "messages": [
        {"role": "system", "content": _sysmsg(
            [("clean-one", "does a thing"), ("dirty", f"{SYNTH} worked example")])},
        {"role": "user", "content": "hello"}]})
    assert r["allowed"]
    body = str(r["api_kwargs"])
    assert SYNTH not in body
    assert "clean-one" in body, "an innocent neighbouring entry must survive"
    assert r["excluded_components"] == {"skills_index": 1}


def test_contaminated_memory_entry_is_dropped_but_neighbours_survive(guard):
    r = _run({"model": "m", "messages": [
        {"role": "system", "content": _sysmsg(
            [("a", "clean")], memory=["keep this one", f"note about {SYNTH}"])},
        {"role": "user", "content": "hello"}]})
    assert r["allowed"]
    body = str(r["api_kwargs"])
    assert SYNTH not in body
    assert "keep this one" in body
    assert r["excluded_components"] == {"memory": 1}


def test_memory_filtering_does_not_swallow_the_prompt_tail(guard):
    """A match in the LAST memory block must not delete what follows it."""
    sep = "═" * 46
    content = (f"{sep}\nUSER PROFILE (who the user is) [1% — 1/100 chars]\n{sep}\n"
               f"profile mentions {SYNTH}\n\n"
               "Conversation started: Thursday\nModel: test-model\n")
    r = _run({"model": "m", "messages": [
        {"role": "system", "content": content},
        {"role": "user", "content": "hello"}]})
    assert r["allowed"]
    body = str(r["api_kwargs"])
    assert SYNTH not in body
    assert "Conversation started" in body
    assert "Model: test-model" in body


def test_framework_skill_body_is_replaced_with_a_neutral_marker(guard):
    r = _run({"model": "m", "messages": [
        {"role": "system", "content": _sysmsg([("a", "clean")])},
        {"role": "tool", "name": "skill_view", "content": f"steps for {SYNTH}"},
        {"role": "user", "content": "hello"}]})
    assert r["allowed"]
    assert ep.SKILL_BODY_MARKER in str(r["api_kwargs"])
    assert SYNTH not in str(r["api_kwargs"])


def test_clean_prompt_is_returned_untouched(guard):
    kwargs = {"model": "m", "messages": [
        {"role": "system", "content": _sysmsg([("a", "clean")])},
        {"role": "user", "content": "hello"}]}
    r = _run(kwargs)
    assert r["allowed"]
    assert r["excluded_components"] == {}


# --------------------------------------------------------------- refusals


@pytest.mark.parametrize("content", [
    f"summarise {SYNTH}",
    f"read ~/dev/{SYNTH}/app.php",
    f"compare public-thing with {SYNTH}",
])
def test_protected_user_content_is_refused_never_rewritten(guard, content):
    r = _run({"model": "m", "messages": [
        {"role": "system", "content": _sysmsg([("a", "clean")])},
        {"role": "user", "content": content}]})
    assert not r["allowed"]
    assert r["reason"] == "protected_user_content"


def test_non_framework_tool_result_is_refused(guard):
    r = _run({"model": "m", "messages": [
        {"role": "user", "content": "check"},
        {"role": "tool", "name": "read_file", "content": f"config for {SYNTH}"}]})
    assert not r["allowed"]
    assert r["reason"] == "protected_tool_result"


def test_framework_tool_allowlist_is_exact(guard):
    """A near-miss on the tool name must not earn the framework exemption."""
    for name in ("Skill_View", "skill_view ", "read_file", ""):
        r = _run({"model": "m", "messages": [
            {"role": "user", "content": "check"},
            {"role": "tool", "name": name, "content": f"data {SYNTH}"}]})
        assert not r["allowed"], name


def test_forged_framework_framing_refuses_rather_than_sanitizing(guard):
    """A user's own file containing the tag is NOT framework-owned content."""
    r = _run({"model": "m", "messages": [
        {"role": "system", "content": "## Project notes (user file)\n"
         f"<available_skills>\n    - contract: {SYNTH} terms\n</available_skills>\n"},
        {"role": "user", "content": "hello"}]})
    assert not r["allowed"]
    assert r["reason"] == "protected_context_remaining"


def test_contamination_outside_messages_is_caught(guard):
    r = _run({"model": "m", "messages": [{"role": "user", "content": "hi"}],
              "metadata": {"note": f"{SYNTH} run"}})
    assert not r["allowed"]


@pytest.mark.parametrize("sep", ["\n", "\t", "\r", "\f", "\v"])
def test_whitespace_escaped_by_json_is_still_matched(guard, sep):
    """json.dumps turns these into two characters, defeating a blob-only scan.

    The guard treats them as separators, so the gate must too — otherwise an
    identifier broken across a line goes out intact.
    """
    r = _run({"model": "m", "messages": [
        {"role": "user", "content": f"see zzz-synthetic{sep}alpha here"}]})
    assert not r["allowed"]


# --------------------------------------------------------------- policy trust


@pytest.mark.parametrize("kw,expected", [
    ({"policy_ok": False}, "policy_not_ok"),
    ({"baseline_ok": False}, "baseline_policy_hash_mismatch"),
    ({"min_rules": 99}, "below_min_rules"),
])
def test_untrusted_policy_fails_closed(monkeypatch, kw, expected):
    monkeypatch.setattr(ep, "_load_guard", lambda: _stub_guard(**kw))
    ep._guard_cache.clear()
    assert ep.policy_state()["reason"] == expected
    r = _run({"model": "m", "messages": [{"role": "user", "content": "totally clean"}]})
    assert not r["allowed"]
    assert r["reason"].startswith("policy_unavailable")


def test_unreadable_guard_fails_closed(monkeypatch):
    monkeypatch.setattr(ep, "GUARD_PATH", "/nonexistent/guard.py")
    ep._guard_cache.clear()
    r = _run({"model": "m", "messages": [{"role": "user", "content": "clean"}]})
    assert not r["allowed"]


def test_guard_path_is_not_environment_configurable():
    """An env-substitutable policy engine would be a complete bypass."""
    assert ep.GUARD_PATH.startswith("/Library/Application Support/EliteProtected")
    assert ep.REQUIRE_ROOT_GUARD is True


# --------------------------------------------------------------- targets


@pytest.fixture
def _routes(monkeypatch):
    """Pin the approved-route set so these tests do not depend on the live config."""
    monkeypatch.setattr(ep, "_configured_local_routes", lambda: [
        ("100.100.175.79", "4000"),                       # configured LiteLLM
        ("127.0.0.1", "11434"), ("localhost", "11434"),   # configured Ollama
        ("127.0.0.1", "1234"), ("localhost", "1234"),     # configured LM Studio
        ("10.10.20.20", "8000"),                          # a configured local vLLM
    ])


@pytest.mark.parametrize("provider,url,external", [
    # The sanctioned gateway IS the egress path, so it is external by definition.
    ("elite-gateway", "http://100.100.175.79:8082/v1", True),
    ("elite-gateway", "http://127.0.0.1:8082/v1", True),      # gateway even on loopback
    ("x", "http://100.100.175.79:8082/v1", True),             # port, not the name
    # CONFIGURED local inference may skip projection.
    ("litellm", "http://100.100.175.79:4000/v1", False),
    ("ollama", "http://localhost:11434/v1", False),
    ("ollama", "http://127.0.0.1:11434/v1", False),
    ("lmstudio", "http://127.0.0.1:1234/v1", False),
    ("vllm", "http://10.10.20.20:8000/v1", False),
    # NOT configured -> projected, whatever the address SHAPE.
    ("x", "http://192.168.1.50:8000/v1", True),               # arbitrary RFC1918
    ("x", "http://10.9.9.9:1234/v1", True),                   # arbitrary RFC1918
    ("x", "http://172.16.3.3:8080/v1", True),                 # arbitrary RFC1918
    ("x", "http://100.74.115.94:9999/v1", True),              # arbitrary tailnet peer
    ("litellm", "http://127.0.0.1:4000/v1", True),            # loopback, unapproved port
    ("x", "http://some-box:11434/v1", True),                  # arbitrary private name
    # Public providers and spoofs.
    ("ollama", "https://api.openai.com/v1", True),            # name is not evidence
    ("x", "https://gw.remote.example:4000/v1", True),         # local port, remote host
    ("x", "https://evil.example/proxy:4000/v1", True),        # :4000 only in the path
    ("x", "http://127.0.0.1.evil.com/v1", True),              # host-prefix spoof
    ("x", "http://127.0.0.1:11434@evil.example.com/v1", True),  # userinfo smuggling
    ("mystery", "https://api.unknown.example/v1", True),      # unknown -> external
    ("x", "", True),                                          # missing -> project
    ("x", "not a url", True),                                 # ambiguous -> project
])
def test_target_classification(_routes, provider, url, external):
    assert ep.is_external_target(provider, url) is external


def test_locality_is_configured_not_inferred_from_address_shape(_routes):
    """A private address is not evidence of an approved local model.

    Treating RFC1918/CGNAT shape as "local" let ANY private endpoint skip
    projection — including an arbitrary peer on the tailnet, which is exactly
    the reachable-and-untrusted case.
    """
    assert ep.is_external_target("x", "http://100.100.175.79:4000/v1") is False
    # same port, unconfigured host:
    assert ep.is_external_target("x", "http://100.74.115.94:4000/v1") is True
    # same host, unconfigured port:
    assert ep.is_external_target("x", "http://100.100.175.79:4001/v1") is True


def test_configured_routes_are_read_from_hermes_config(monkeypatch):
    """The route set comes from Hermes' own config — no JOB D dependency."""
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config_readonly", lambda: {
        "providers": {
            "litellm": {"base_url": "http://100.100.175.79:4000/v1"},
            "elite-gateway": {"base_url": "http://100.100.175.79:8082/v1"},
            "openai": {"base_url": "https://api.openai.com/v1"},
            "portless": {"base_url": "http://10.0.0.9"},
        }}, raising=False)
    routes = ep._configured_local_routes()
    assert ("100.100.175.79", "4000") in routes            # configured local
    assert ("100.100.175.79", "8082") not in routes        # the gateway is egress
    assert not any(h == "api.openai.com" for h, _ in routes)   # public excluded
    assert not any(h == "10.0.0.9" for h, _ in routes)     # portless approves nothing


def test_module_carries_no_job_d_dependency():
    """JOB C lands first and must stand alone."""
    import pathlib
    assert "aux_egress_policy" not in pathlib.Path(ep.__file__).read_text()


def test_configured_local_target_is_returned_completely_untouched(guard, _routes):
    kwargs = {"model": "m", "messages": [
        {"role": "system", "content": _sysmsg([("a", f"{SYNTH} notes")])},
        {"role": "user", "content": f"work on {SYNTH}"}]}
    r = ep.guard_outbound(kwargs, provider="litellm",
                          base_url="http://100.100.175.79:4000/v1", request_id="t")
    assert r["allowed"] and r["external"] is False
    assert r["api_kwargs"] is kwargs, "local requests must not be copied or filtered"


def test_unconfigured_private_target_is_projected_not_trusted(guard, _routes):
    """The identical payload to an UNAPPROVED private endpoint must not pass raw."""
    kwargs = {"model": "m", "messages": [
        {"role": "system", "content": _sysmsg([("a", f"{SYNTH} notes")])},
        {"role": "user", "content": f"work on {SYNTH}"}]}
    r = ep.guard_outbound(kwargs, provider="litellm",
                          base_url="http://192.168.1.50:4000/v1", request_id="t")
    assert r["external"] is True
    assert not r["allowed"], "protected user content must be refused, never rewritten"
    assert SYNTH not in str(r.get("api_kwargs") or "")


def test_mock_shaped_attributes_do_not_raise():
    """These come off the agent object, which is a Mock in much of the suite."""
    class Weird:
        def __str__(self):
            raise RuntimeError("nope")

    assert ep.is_external_target(None, None) is True
    assert ep.is_external_target(object(), object()) is True


# --------------------------------------------------------------- hygiene


def test_audit_and_metrics_carry_no_content(guard, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    ep.COUNTERS.clear()
    _run({"model": "m", "messages": [
        {"role": "system", "content": _sysmsg([("a", f"{SYNTH} x")])},
        {"role": "user", "content": "hello"}]})
    for line in ep.metrics_text():
        assert SYNTH not in line
        for banned in ("project_id", "path=", "prompt", "filename", "user="):
            assert banned not in line


# --------------------------------------------------------------- traversal

# Captured at import time, BEFORE tests/conftest.py's autouse
# ``_elite_projection_offline`` fixture swaps it for a pass-through.
from agent import chat_completion_helpers as _cch  # noqa: E402
_REAL_GUARD_OUTBOUND = _cch._elite_guard_outbound

# A real newline, not the two characters backslash+n. json.dumps() rewrites it
# as an ESCAPE, which the guard's normalize() no longer collapses to a
# separator — so the serialized-blob scan cannot see this identifier and the
# raw-leaf walk is the only gate that can.
SPLIT_SYNTH = "zzz-synthetic-\nalpha"


def _nest(value, depth):
    node = value
    for _ in range(depth):
        node = {"n": node}
    return node


@pytest.mark.parametrize("depth", [5, 39, 41, 60, 200])
def test_deep_nesting_cannot_escape_the_leaf_scan(guard, depth):
    """The proven bypass: split the identifier with real whitespace AND nest it.

    ``_iter_strings`` used to ``return`` silently once recursion passed depth 40,
    so every leaf below that was reported clean rather than unscanned — the
    wrong failure direction for a security gate. The sibling blob scan could not
    compensate, because JSON escaping hides the whitespace split from
    ``normalize()``. Both gates therefore missed this payload together.

    Depths 5 and 39 are included so the test still fails if someone "fixes" this
    by raising the cap instead of removing it.
    """
    r = _run({"model": "m",
              "messages": [{"role": "user", "content": "hello"}],
              "meta": _nest(SPLIT_SYNTH, depth)})
    assert not r["allowed"], f"a protected identifier at depth {depth} escaped"
    assert "synthetic" not in str(r.get("api_kwargs") or "")


@pytest.mark.parametrize("depth", [5, 41, 200])
def test_clean_deeply_nested_payload_is_still_allowed(guard, depth):
    """Removing the cap must not turn deep nesting into a blanket refusal."""
    r = _run({"model": "m",
              "messages": [{"role": "user", "content": "hello"}],
              "meta": _nest("perfectly ordinary text", depth)})
    assert r["allowed"]


def test_traversal_reaches_a_leaf_far_below_the_old_cap(guard):
    assert "needle-at-the-bottom" in list(ep._iter_strings(
        _nest("needle-at-the-bottom", 500)))


def test_dict_keys_are_scanned_not_only_values(guard):
    r = _run({"model": "m",
              "messages": [{"role": "user", "content": "hello"}],
              "meta": _nest({SPLIT_SYNTH: "value"}, 50)})
    assert not r["allowed"]


def test_unstringifiable_object_fails_closed(guard):
    """If we cannot see what would be sent, refuse — never skip and report clean."""
    class _Opaque:
        def __str__(self):
            raise RuntimeError("cannot render")

    r = _run({"model": "m",
              "messages": [{"role": "user", "content": "hello"}],
              "x": _Opaque()})
    assert not r["allowed"]
    assert r["reason"].startswith("unscannable_payload"), r["reason"]


def test_cyclic_structure_fails_closed(guard):
    """A cycle must terminate the walk by REFUSING, not by looping or skipping."""
    cyc = {"note": "hi"}
    cyc["self"] = cyc
    r = _run({"model": "m",
              "messages": [{"role": "user", "content": "hello"}],
              "x": cyc})
    assert not r["allowed"]


def test_cycle_does_not_hide_a_sibling_leaf(guard):
    """Deduplicating visited containers must not stop the scan reaching siblings."""
    inner = {"deep": SPLIT_SYNTH}
    payload = {"model": "m",
               "messages": [{"role": "user", "content": "hello"}],
               "a": inner, "b": inner}
    assert not _run(payload)["allowed"]


def test_scan_budget_raises_rather_than_truncating(monkeypatch):
    """A pathological payload must refuse, never silently stop scanning."""
    monkeypatch.setattr(ep, "_SCAN_NODE_BUDGET", 5)
    with pytest.raises(ep.UnscannablePayload):
        list(ep._iter_strings(_nest("x", 50)))


# ------------------------------------------------------- choke points


class _FakeAgent:
    provider = "elite-gateway"
    base_url = GATEWAY
    session_id = "test-session"
    api_mode = "chat_completions"
    _interrupt_requested = False


@pytest.mark.parametrize("payload", [
    {"model": "m", "messages": [{"role": "user", "content": f"about {SYNTH}"}]},
    {"model": "m", "messages": [{"role": "user", "content": "hi"}],
     "meta": _nest(SPLIT_SYNTH, 60)},
])
@pytest.mark.parametrize("entry", ["interruptible_api_call",
                                   "interruptible_streaming_api_call"])
def test_no_provider_call_on_refusal_at_either_choke_point(
        guard, monkeypatch, entry, payload):
    """A refusal must raise before anything can be put on the wire.

    Both entry points call the gate as their FIRST statement, so a worker thread
    is never started and no client is ever constructed.
    """
    import threading
    monkeypatch.setattr(_cch, "_elite_guard_outbound", _REAL_GUARD_OUTBOUND)
    starts = []
    real_start = threading.Thread.start
    monkeypatch.setattr(threading.Thread, "start",
                        lambda self, *a, **k: starts.append(self) or real_start(self))

    with pytest.raises(_cch.ExternalProjectionBlocked) as exc:
        getattr(_cch, entry)(_FakeAgent(), payload)

    assert starts == [], "a refused request started a request worker"
    assert "NOT sent to any external provider" in str(exc.value)
    assert exc.value.status_code == 400, "401/403 would drain the credential pool"


def test_clean_external_request_passes_the_choke_point(guard, monkeypatch):
    """The gate must not become a blanket refusal for ordinary work."""
    monkeypatch.setattr(_cch, "_elite_guard_outbound", _REAL_GUARD_OUTBOUND)
    out = _REAL_GUARD_OUTBOUND(
        _FakeAgent(), {"model": "m",
                       "messages": [{"role": "user", "content": "refactor this loop"}]})
    assert out["messages"][0]["content"] == "refactor this loop"


# ------------------------------------------- configured-route admission


def _with_config(monkeypatch, providers):
    """Drive _configured_local_routes from a synthetic Hermes config."""
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config_readonly",
                        lambda: {"providers": providers}, raising=False)


@pytest.mark.parametrize("label,url,expect_local", [
    # 1. a configured ARBITRARY PUBLIC hostname must not become an approved route
    ("arbitrary public host",  "https://evil.example.com:4000",             False),
    # 2. a configured UNKNOWN provider hostname — absent from any deny list
    ("unknown public provider", "https://api.some-new-provider.example:4000", False),
    ("unknown public provider", "https://inference.brand-new-ai.io:11434",   False),
    # 3. configured RFC1918 exact host+port
    ("configured RFC1918",     "http://192.168.1.50:4000",                  True),
    ("configured RFC1918",     "http://10.9.9.9:8000",                      True),
    ("configured RFC1918",     "http://172.16.3.3:8080",                    True),
    # 4. configured Tailscale/CGNAT exact host+port
    ("configured tailnet",     "http://100.100.175.79:4000",                True),
    ("configured tailnet",     "http://100.74.115.94:8000",                 True),
    # 6. the gateway is egress, never an approved route
    ("gateway 8082",           "http://100.100.175.79:8082",                False),
    # 7. approved loopback service name
    ("localhost service",      "http://localhost:11434",                    True),
    ("loopback literal",       "http://127.0.0.1:4000",                     True),
    # address classes that are private-shaped but are not evidence of locality
    ("link-local metadata",    "http://169.254.169.254:4000",               False),
    ("bind-all address",       "http://0.0.0.0:4000",                       False),
    # a DNS name is never evidence, however innocent it looks
    ("plausible local name",   "http://ollama.internal:11434",              False),
    ("loopback-prefixed name", "http://127.0.0.1.evil.com:11434",           False),
])
def test_configured_route_admission(monkeypatch, label, url, expect_local):
    """Configuration is PERMISSION to skip projection, never EVIDENCE of locality.

    A route has to be declared AND name an address that cannot be a public
    provider. Admitting on "declared, and not on a known-public list" would
    leave every new or unlisted provider approved by default — the wrong
    default for a gate whose job is to decide what leaves the machine.
    """
    _with_config(monkeypatch, {"p": {"base_url": url}})
    host, port = ep._host_port(url)
    admitted = (host, port) in ep._configured_local_routes()
    assert admitted is expect_local, f"{label}: {url}"
    # and the classification that rides on it
    assert ep.is_external_target("p", url) is (not expect_local)


def test_configured_private_host_on_the_wrong_port_is_external(monkeypatch):
    """5. Admission is per (host, port), never per host."""
    _with_config(monkeypatch, {"p": {"base_url": "http://192.168.1.50:4000"}})
    assert ep.is_external_target("p", "http://192.168.1.50:4000/v1") is False
    assert ep.is_external_target("p", "http://192.168.1.50:4001/v1") is True
    assert ep.is_external_target("p", "http://192.168.1.50:8000/v1") is True


def test_public_host_list_is_not_the_load_bearing_check(monkeypatch):
    """An unlisted public provider must stay external without being enumerated."""
    url = "https://api.brand-new-provider-2030.example:4000"
    host, _ = ep._host_port(url)
    assert not any(pub in host for pub in ep._PUBLIC_PROVIDER_HOSTS), (
        "this test is only meaningful for a host the deny list does NOT name")
    _with_config(monkeypatch, {"p": {"base_url": url}})
    assert ep.is_external_target("p", url) is True


def test_a_configured_public_route_cannot_carry_protected_content_raw(guard, monkeypatch):
    """End to end: the config edit must not switch projection off."""
    _with_config(monkeypatch, {"p": {"base_url": "https://evil.example.com:4000"}})
    kwargs = {"model": "m", "messages": [
        {"role": "user", "content": f"work on {SYNTH}"}]}
    r = ep.guard_outbound(kwargs, provider="p",
                          base_url="https://evil.example.com:4000/v1", request_id="t")
    assert r["external"] is True
    assert not r["allowed"]
    assert SYNTH not in str(r.get("api_kwargs") or "")


@pytest.mark.parametrize("host,local", [
    ("127.0.0.1", True), ("127.1.2.3", True), ("::1", True), ("localhost", True),
    ("10.0.0.1", True), ("172.16.0.1", True), ("192.168.0.1", True),
    ("100.64.0.1", True), ("100.127.255.254", True),        # CGNAT bounds
    ("100.63.255.255", False), ("100.128.0.0", False),      # just outside CGNAT
    ("169.254.169.254", False),                             # cloud metadata
    ("0.0.0.0", False), ("::", False), ("255.255.255.255", False),
    ("2002:c000:204::1", False),                            # 6to4 embeds 192.0.2.4
    ("2001:0:c000:204::", False),                           # teredo
    ("evil.example.com", False), ("some-box", False), ("", False),
])
def test_is_local_literal(host, local):
    assert ep._is_local_literal(host) is local
