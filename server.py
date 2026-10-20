# server.py — opencode-gateway as a local Python API (Flask).
#
# Same wire surface as worker.js. Lets any external harness (Claude Code,
# Codex CLI, Aider, LangChain, ...) use OpenCode models through one
# OpenAI/Anthropic-compatible endpoint.
#
# TWO MODES:
#
# 1. FREE MODE (no key, or Authorization: Bearer public)
#    Uses OpenCode's free model catalog via the same client-shape fingerprint
#    the official opencode CLI presents (reverse-engineered and maintained by
#    the opencode2api project):
#      - POST /inference/{openai}/v1/... (not /zen/v1)
#      - User-Agent: opencode/latest/<ver>/cli
#      - x-opencode-session: ses_ + descending 48-bit time hex + 14 base62
#      - body.stream must be true (non-stream clients get local aggregation)
#      - tools must include bash/glob/grep/read with non-empty properties
#      - x-opencode-request must NOT be sent; x-opencode-project: global
#    Upstream may tighten these checks any time; env OC_UA_VERSION re-pins
#    the UA version.
#
# 2. KEYED MODE (real Zen key via X-Zen-Key / Bearer / x-api-key / ?key=,
#    or env ZEN_API_KEY)
#    Transparent proxy to https://opencode.ai/zen/v1/... — no body changes.
#
# Routes (client-facing, always under /v1):
#   GET  /v1/models            catalog (free models annotated)
#   POST /v1/chat/completions  OpenAI Chat wire (stream + non-stream)
#   POST /v1/messages          Anthropic Messages wire (stream + non-stream)
#   POST /v1/responses         OpenAI Responses wire (stream + non-stream)
#   POST /v1/systemone         Jev/SystemOne decision wire (free tier)
#   GET  /health
#
# Optional client auth: set GATEWAY_KEY env; callers must then present it
# via Authorization: Bearer <gateway> or X-Gateway-Key.
#
# Run:  python server.py
#       gunicorn -w 4 -b 0.0.0.0:8080 server:app
#
# Deps: flask + httpx.

import json
import os
import secrets
import time

import httpx
from flask import Flask, Response, request

from bridge import (anthropic_to_chat_body, responses_to_chat_body, chat_to_anthropic,
                    chat_to_responses, sse_from_anthropic_message, sse_from_responses_object,
                    aggregate_chat_sse, aggregate_responses_sse, parse_sse_text)

UPSTREAM = "https://opencode.ai"
OC_UA_VERSION = "2.0.29"  # pinned version the gate accepts (verified live)
KNOWN_FREE = {"big-pickle"}
STUB_NAMES = ["bash", "glob", "grep", "read"]
STUB_PROPS = {"bash": "command", "glob": "pattern", "grep": "pattern", "read": "file_path"}

PROXY_PATHS = {"/v1/chat/completions", "/v1/messages", "/v1/responses",
               "/v1/completions", "/v1/embeddings"}
# Jev/SystemOne is a TypeSafe DECISION model (state + questions -> answers
# with probabilities), not a chat model — it has its own wire format and
# endpoint. Free-tier verified: POST /inference/systemone/v1/systemone with
# the standard fingerprint headers, plain JSON body (no stream/tools gate).
SYSTEMONE_PATH = "/v1/systemone"
SYSTEMONE_UPSTREAM = "/inference/systemone/v1/systemone"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers":
        "Authorization, Content-Type, X-Zen-Key, X-Gateway-Key, X-Api-Key, Anthropic-Version, Anthropic-Beta",
    "Access-Control-Expose-Headers": "X-Gateway-Mode, X-Gateway-Upstream-Status",
    "Access-Control-Max-Age": "86400",
}

_client = httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))

app = Flask(__name__)


# ---------- OpenCode ID shape (packages/opencode/src/id/id.ts) ----------
# <prefix>_ + 12 lowercase hex (48-bit time component, descending for
# sessions) + 14 base62 chars. time = ms*0x1000 + small counter.
_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _rand_b62(n):
    return "".join(_B62[b % 62] for b in secrets.token_bytes(n))


def oc_session_id():
    ms = int(time.time() * 1000)
    counter = 1 + secrets.randbelow(8)  # keep <= 48
    v = (ms << 12) + counter
    v = ~v  # descending (sessions)
    v &= (1 << 48) - 1  # low 48 bits
    return "ses_" + format(v, "012x") + _rand_b62(14)


# ---------- free-tier fingerprint ----------

def _stub_tool_chat(name, prop):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {
                "type": "object",
                "properties": {prop: {"type": "string", "description": prop}},
                "required": [prop],
            },
        },
    }


def _stub_tool_anthropic(name, prop):
    return {
        "name": name,
        "description": f"{name} tool",
        "input_schema": {
            "type": "object",
            "properties": {prop: {"type": "string", "description": prop}},
            "required": [prop],
        },
    }


def _tool_name_of(t):
    if isinstance(t, dict):
        fn = t.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return fn["name"]
        if isinstance(t.get("name"), str):
            return t["name"]
    return ""


def inject_tools(body, wire):
    """Inject missing required tools; returns map {stubLowerName ->
    clientOriginal} for response-side case restore (Claude Code registers
    PascalCase tools)."""
    existing = set()
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    for t in tools:
        n = _tool_name_of(t)
        if n and n == n.lower():
            existing.add(n)
    restore = {}
    for t in tools:
        n = _tool_name_of(t)
        if n and n.lower() in STUB_NAMES and n != n.lower():
            restore[n.lower()] = n
    missing = []
    for name in STUB_NAMES:
        if name in existing:
            continue
        prop = STUB_PROPS[name]
        if wire in ("messages", "responses"):
            missing.append(_stub_tool_anthropic(name, prop))
        else:
            missing.append(_stub_tool_chat(name, prop))
    if missing:
        body["tools"] = tools + missing
    if not tools:
        # client brought no tools: forbid the model from calling our stubs
        body["tool_choice"] = {"type": "none"} if wire == "messages" else "none"
    return restore


def is_free_model_id(model_id):
    return model_id.endswith("-free") or model_id in KNOWN_FREE


# ---------- name restore (tool-name case) ----------

def restore_names_deep(obj, restore):
    if not isinstance(obj, dict) or not restore:
        return obj
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k == "name" and isinstance(v, str) and v in restore:
                obj[k] = restore[v]
            else:
                restore_names_deep(v, restore)
    elif isinstance(obj, list):
        for v in obj:
            restore_names_deep(v, restore)
    return obj


def _restore_in_sse_line(line, restore):
    """Rewrite one SSE data line with tool-name restore (streaming passthrough)."""
    if not line.startswith("data:"):
        return line
    nl = line.endswith("\n")
    payload = line[5:].strip() if line[5:].startswith(" ") else line[5:]
    payload = payload.strip()
    if not payload or payload == "[DONE]":
        return line
    try:
        j = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return line
    before = json.dumps(j, separators=(",", ":"))
    restore_names_deep(j, restore)
    if json.dumps(j, separators=(",", ":")) != before:
        return "data: " + json.dumps(j, separators=(",", ":")) + ("\n" if nl else "")
    return line


# ---------- key resolution ----------

def resolve_keys():
    auth = request.headers.get("authorization") or ""
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    gw = (os.environ.get("GATEWAY_KEY") or "").strip()

    zen_key = (
        (request.headers.get("x-zen-key") or "").strip()
        or (request.args.get("key") or "").strip()
        or (bearer and not (gw and bearer == gw) and bearer or "")
        or (request.headers.get("x-api-key") or "").strip()
        or ""
    )

    if gw:
        client_key = (request.headers.get("x-gateway-key") or "").strip() or (bearer if bearer == gw else "")
        if not client_key:
            return {"error": _json({"error": {"type": "auth_error",
                    "message": "Missing gateway key. Send 'Authorization: Bearer <GATEWAY_KEY>' or 'X-Gateway-Key'."}}, 401)}
        if client_key != gw:
            return {"error": _json({"error": {"type": "auth_error",
                    "message": "Invalid gateway key."}}, 401)}

    if not zen_key:
        zen_key = (os.environ.get("ZEN_API_KEY") or "").strip()
    # 'public' or empty -> free tier
    free = not zen_key or zen_key == "public"
    return {"zen_key": None if free else zen_key, "free": free}


# ---------- helpers ----------

def _json(data, status=200, extra=None):
    headers = {"content-type": "application/json", **CORS_HEADERS, **(extra or {})}
    return Response(json.dumps(data), status=status, headers=headers)


def _env_version():
    return (os.environ.get("OC_UA_VERSION") or OC_UA_VERSION).strip()


# ---------- SystemOne (Jev decision wire, free tier) ----------

def handle_systemone():
    session = oc_session_id()
    headers = {
        "content-type": "application/json",
        "authorization": "Bearer public",
        "user-agent": f"opencode/latest/{_env_version()}/cli",
        "x-opencode-client": "cli",
        "x-opencode-project": "global",
        "x-opencode-session": session,
        "x-session-id": session,
        "x-session-affinity": session,
        "x-opencode-session-id": session,
        "accept": "application/json",
    }
    raw = request.get_data()
    try:
        resp = _client.post(UPSTREAM + SYSTEMONE_UPSTREAM, headers=headers, content=raw)
    except httpx.HTTPError as exc:
        return _json({"error": {"type": "upstream_error",
                                "message": f"Upstream fetch failed: {exc}"}}, 502)
    return Response(resp.content, status=resp.status_code, headers={
        "content-type": resp.headers.get("content-type") or "application/json",
        **CORS_HEADERS,
        "x-gateway-mode": "free",
        "x-gateway-upstream-status": str(resp.status_code),
    })


def handle_models():
    try:
        r = _client.get(UPSTREAM + "/zen/v1/models")
        data = r.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return _json({"error": {"type": "upstream_error", "message": str(exc)}}, 502)
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for m in data["data"]:
            if isinstance(m, dict):
                m["free"] = is_free_model_id(m.get("id") or "")
    return _json(data, r.status_code)


# ---------- free mode ----------

def handle_free(body_json, wire):
    body = body_json
    if not isinstance(body, dict) or not body.get("model"):
        return _json({"error": {"type": "invalid_request",
                                "message": "'model' is required."}}, 400)
    model = body["model"]
    if not is_free_model_id(model):
        return _json({
            "error": {
                "type": "not_free_model",
                "message": f"Model '{model}' is not in the free catalog. Free models: see GET /v1/models (free:true). "
                           "For paid models pass a Zen key (https://opencode.ai/auth) via Authorization/X-Zen-Key.",
            },
        }, 400)

    client_stream = body.get("stream") is True

    # Bridge the client wire to the upstream chat wire.
    restore = {}
    if wire == "messages":
        chat_body = anthropic_to_chat_body(body)
    elif wire == "responses":
        chat_body = responses_to_chat_body(body)
    else:
        chat_body = body
        restore = inject_tools(chat_body, "chat")

    chat_body["model"] = model
    chat_body["stream"] = True  # upstream gate: stream must be true
    chat_body["stream_options"] = {**(chat_body.get("stream_options") or {}), "include_usage": True}
    # Always ensure the 4 required tools exist (adds missing stubs; sets
    # tool_choice:none only when the client brought no tools at all).
    restore = inject_tools(chat_body, "chat")

    session = oc_session_id()
    headers = {
        "content-type": "application/json",
        "authorization": "Bearer public",
        "user-agent": f"opencode/latest/{_env_version()}/cli",
        "x-opencode-client": "cli",
        "x-opencode-project": "global",
        "x-opencode-session": session,
        "x-session-id": session,
        "x-session-affinity": session,
        "x-opencode-session-id": session,
        "accept": "text/event-stream",
    }
    # NOTE: x-opencode-request must NOT be sent on /inference/* paths (403).

    upstream_url = f"{UPSTREAM}/inference/openai/v1/chat/completions"
    try:
        resp = _client.post(upstream_url, headers=headers,
                            content=json.dumps(chat_body).encode())
    except httpx.HTTPError as exc:
        return _json({"error": {"type": "upstream_error",
                                "message": f"Upstream fetch failed: {exc}"}}, 502)

    if resp.status_code >= 400:
        return Response(resp.content, status=resp.status_code, headers={
            "content-type": resp.headers.get("content-type") or "application/json",
            **CORS_HEADERS,
            "x-gateway-mode": "free",
            "x-gateway-upstream-status": str(resp.status_code),
        })

    raw = resp.content.decode("utf-8", "replace")

    if client_stream:
        # Native chat wire: stream upstream chunks straight through (with
        # tool-name restore). Bridged wires: buffer + translate so the client
        # gets correct Anthropic/Responses SSE events.
        if wire == "chat":
            if restore:
                body_lines = "".join(_restore_in_sse_line(ln + "\n", restore)
                                     for ln in raw.split("\n"))
                return Response(body_lines, status=200, headers={
                    "content-type": "text/event-stream", "cache-control": "no-store",
                    **CORS_HEADERS, "x-gateway-mode": "free",
                })
            return Response(raw, status=200, headers={
                "content-type": "text/event-stream", "cache-control": "no-store",
                **CORS_HEADERS, "x-gateway-mode": "free",
            })
        agg = aggregate_chat_sse(raw, model)
        restore_names_deep(agg, restore)
        if not agg:
            return _json({"error": {"type": "upstream_error",
                                    "message": "Could not translate upstream stream.",
                                    "raw": raw[:400]}}, 502, {"x-gateway-mode": "free"})
        sse_body = (sse_from_anthropic_message(chat_to_anthropic(agg))
                    if wire == "messages"
                    else sse_from_responses_object(chat_to_responses(agg)))
        return Response(sse_body, status=200, headers={
            "content-type": "text/event-stream", "cache-control": "no-store",
            **CORS_HEADERS, "x-gateway-mode": "free",
        })

    # client wanted non-streaming: aggregate
    agg = aggregate_chat_sse(raw, model)
    restore_names_deep(agg, restore)
    if not agg:
        return _json({"error": {"type": "upstream_error",
                                "message": "Could not aggregate upstream stream into a non-streaming response.",
                                "raw": raw[:400]}}, 502, {"x-gateway-mode": "free"})
    if wire == "messages":
        return _json(chat_to_anthropic(agg), 200, {"x-gateway-mode": "free"})
    if wire == "responses":
        return _json(chat_to_responses(agg), 200, {"x-gateway-mode": "free"})
    return _json(agg, 200, {"x-gateway-mode": "free"})


# ---------- keyed mode ----------

def handle_keyed(path, zen_key, body_text, body_json):
    upstream_url = UPSTREAM + "/zen" + path + ("?" + request.query_string.decode("latin-1")
                                               if request.query_string else "")
    headers = {"content-type": "application/json",
               "authorization": f"Bearer {zen_key}",
               "x-api-key": zen_key}
    av = request.headers.get("anthropic-version")
    if av:
        headers["anthropic-version"] = av
    ab = request.headers.get("anthropic-beta")
    if ab:
        headers["anthropic-beta"] = ab

    try:
        req = _client.build_request(request.method, upstream_url, headers=headers,
                                    content=body_text.encode() if body_text is not None else None)
        resp = _client.send(req, stream=True)
    except httpx.HTTPError as exc:
        return _json({"error": {"type": "upstream_error",
                                "message": f"Upstream fetch failed: {exc}"}}, 502)

    # Free-model fallback: harnesses (Hermes, OpenAI SDKs) always send an
    # Authorization header — often a placeholder. If keyed mode fails on a
    # model we can serve for free, retry through the free fingerprint.
    model = (body_json or {}).get("model") if isinstance(body_json, dict) else None
    bad_key = resp.status_code in (401, 403) or resp.status_code >= 500
    if bad_key and model and is_free_model_id(model) and \
            path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
        resp.close()
        return handle_free(body_json, wire_for(path))

    out_headers = {"x-gateway-mode": "keyed",
                   "x-gateway-upstream-status": str(resp.status_code),
                   **CORS_HEADERS}
    ct = resp.headers.get("content-type")
    if ct:
        out_headers["content-type"] = ct

    def generate():
        try:
            for chunk in resp.iter_raw():
                yield chunk
        finally:
            resp.close()

    return Response(generate(), status=resp.status_code, headers=out_headers)


def wire_for(path):
    return {"": "chat", "/v1/messages": "messages", "/v1/responses": "responses"}.get(path, "chat")


# ---------- routes ----------

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return _json({
        "ok": True,
        "service": "opencode-gateway (python)",
        "modes": {
            "free": "no key / Bearer public -> OpenCode free models (client-shape fingerprint)",
            "keyed": "Zen key via Authorization/X-Zen-Key/x-api-key/?key= (or ZEN_API_KEY env) -> /zen/v1 proxy",
        },
        "routes": ["/v1/models", "/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/systemone"],
        "pinned_oc_version": _env_version(),
        "gateway_key_required": bool(os.environ.get("GATEWAY_KEY")),
    })


@app.before_request
def _options_short_circuit():
    if request.method == "OPTIONS":
        return Response(status=204)


@app.route("/v1/models", methods=["GET"])
def models():
    return handle_models()


@app.route(SYSTEMONE_PATH, methods=["POST"])
def systemone():
    return handle_systemone()


@app.route("/<path:path>", methods=["POST"])
def proxy(path):
    full = "/" + path
    if full not in PROXY_PATHS:
        return _json({
            "error": {
                "type": "not_found",
                "message": f"Unsupported POST {full}. Use POST /v1/chat/completions, /v1/messages, /v1/responses, or GET /v1/models.",
            },
        }, 404)

    resolved = resolve_keys()
    if resolved.get("error"):
        return resolved["error"]

    body_text = request.get_data(as_text=True)
    try:
        body_json = json.loads(body_text) if body_text else None
    except json.JSONDecodeError:
        body_json = None

    if resolved["free"]:
        wire = wire_for(full)
        if wire in ("chat", "messages", "responses"):
            return handle_free(body_json, wire)
        return _json({"error": {"type": "auth_error",
                                "message": "Free mode supports chat/completions, messages and responses only."}}, 401)
    return handle_keyed(full, resolved["zen_key"], body_text, body_json)


@app.route("/<path:_any>", methods=["GET", "POST"])
def not_found(_any):
    return _json({
        "error": {
            "type": "not_found",
            "message": f"Unsupported {request.method} /{_any}. Use POST /v1/chat/completions, /v1/messages, /v1/responses, or GET /v1/models.",
        },
    }, 404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, threaded=True)
