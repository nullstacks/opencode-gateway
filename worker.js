// opencode-gateway — Cloudflare Worker
// Lets any external harness (Claude Code, Codex CLI, Aider, LangChain, ...)
// use OpenCode models through one OpenAI/Anthropic-compatible endpoint.
//
// TWO MODES:
//
// 1. FREE MODE (no key, or Authorization: Bearer public)
//    Uses OpenCode's free model catalog (big-pickle, *-free, ...) via the
//    same client-shape fingerprint the official opencode CLI presents
//    (reverse-engineered and maintained by the opencode2api project):
//      - POST /inference/{openai|anthropic}/v1/... (not /zen/v1)
//      - User-Agent: opencode/latest/<ver>/cli
//      - x-opencode-session: ses_ + descending 48-bit time hex + 14 base62
//      - body.stream must be true (non-stream clients get local aggregation)
//      - tools must include bash/glob/grep/read with non-empty properties
//      - x-opencode-request must NOT be sent; x-opencode-project: global
//    Upstream may tighten these checks any time (see opencode2api
//    docs/UPSTREAM-COMPAT.md); env OC_UA_VERSION re-pins the UA version.
//
// 2. KEYED MODE (real Zen key via X-Zen-Key / Bearer / x-api-key / ?key=,
//    or env ZEN_API_KEY)
//    Transparent proxy to https://opencode.ai/zen/v1/... — no body changes.
//
// Routes (client-facing, always under /v1):
//   GET  /v1/models            catalog (free models annotated)
//   POST /v1/chat/completions  OpenAI Chat wire (stream + non-stream)
//   POST /v1/messages          Anthropic Messages wire (stream + non-stream)
//   POST /v1/responses         OpenAI Responses wire (stream + non-stream)
//   GET  /health
//
// Optional client auth: set GATEWAY_KEY secret; callers must then present it
// via Authorization: Bearer <gateway> or X-Gateway-Key.

// All free models are served on the OpenAI chat wire upstream; Anthropic and
// Responses client requests are bridged (see bridge.js — same architecture
// as 12errh/zen-proxy).
import { anthropicToChatBody, responsesToChatBody, chatToAnthropic, chatToResponses, sseFromAnthropicMessage, sseFromResponsesObject } from "./bridge.js";

const UPSTREAM = "https://opencode.ai";
const OC_UA_VERSION = "2.0.29"; // pinned version the gate accepts (verified live)
const KNOWN_FREE = new Set(["big-pickle"]);
const STUB_NAMES = ["bash", "glob", "grep", "read"];

const PROXY_PATHS = new Set(["/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/completions", "/v1/embeddings"]);
// Jev/SystemOne is a TypeSafe DECISION model (state + questions -> answers
// with probabilities), not a chat model — it has its own wire format and
// endpoint. Free-tier verified: POST /inference/systemone/v1/systemone with
// the standard fingerprint headers, plain JSON body (no stream/tools gate).
const SYSTEMONE_PATH = "/v1/systemone";
const SYSTEMONE_UPSTREAM = "/inference/systemone/v1/systemone";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers":
    "Authorization, Content-Type, X-Zen-Key, X-Gateway-Key, X-Api-Key, Anthropic-Version, Anthropic-Beta",
  "Access-Control-Expose-Headers": "X-Gateway-Mode, X-Gateway-Upstream-Status",
  "Access-Control-Max-Age": "86400",
};

function json(data, status = 200, extra = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json", ...CORS, ...extra },
  });
}

// ---------- OpenCode ID shape (packages/opencode/src/id/id.ts) ----------
// <prefix>_ + 12 lowercase hex (48-bit time component, descending for
// sessions) + 14 base62 chars. time = ms*0x1000 + small counter.
const B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

function randB62(n) {
  const a = new Uint8Array(n);
  crypto.getRandomValues(a);
  let s = "";
  for (let i = 0; i < n; i++) s += B62[a[i] % 62];
  return s;
}

function ocSessionID() {
  const ms = BigInt(Date.now());
  const counter = BigInt(1 + Math.floor(Math.random() * 8)); // keep <= 48
  let v = (ms << 12n) + counter;
  v = ~v; // descending (sessions)
  v &= 0xffffffffffffn; // low 48 bits
  const hex = v.toString(16).padStart(12, "0");
  return "ses_" + hex + randB62(14);
}

// ---------- free-tier fingerprint ----------
function stubToolChat(name, prop) {
  return {
    type: "function",
    function: {
      name,
      description: `${name} tool`,
      parameters: {
        type: "object",
        properties: { [prop]: { type: "string", description: prop } },
        required: [prop],
      },
    },
  };
}
function stubToolAnthropic(name, prop) {
  return {
    name,
    description: `${name} tool`,
    input_schema: {
      type: "object",
      properties: { [prop]: { type: "string", description: prop } },
      required: [prop],
    },
  };
}
function stubToolResponses(name, prop) {
  return {
    type: "function",
    name,
    description: `${name} tool`,
    parameters: {
      type: "object",
      properties: { [prop]: { type: "string", description: prop } },
      required: [prop],
    },
  };
}
const STUB_PROPS = { bash: "command", glob: "pattern", grep: "pattern", read: "file_path" };

function toolNameOf(t) {
  if (t && typeof t === "object") {
    if (t.function && typeof t.function === "object" && typeof t.function.name === "string")
      return t.function.name;
    if (typeof t.name === "string") return t.name;
  }
  return "";
}

// Inject missing required tools; returns map {stubLowerName -> clientOriginal}
// for response-side case restore (Claude Code registers PascalCase tools).
function injectTools(body, wire) {
  const existing = new Set();
  const tools = Array.isArray(body.tools) ? body.tools : [];
  for (const t of tools) {
    const n = toolNameOf(t);
    if (n && n === n.toLowerCase()) existing.add(n);
  }
  const restore = {};
  for (const t of tools) {
    const n = toolNameOf(t);
    if (n && STUB_NAMES.includes(n.toLowerCase()) && n !== n.toLowerCase())
      restore[n.toLowerCase()] = n;
  }
  const missing = [];
  for (const name of STUB_NAMES) {
    if (existing.has(name)) continue;
    const prop = STUB_PROPS[name];
    if (wire === "messages" || wire === "responses") missing.push(stubToolAnthropic(name, prop));
    else missing.push(stubToolChat(name, prop));
  }
  if (missing.length) body.tools = [...tools, ...missing];
  if (!tools.length) {
    // client brought no tools: forbid the model from calling our stubs
    body.tool_choice = wire === "messages" ? { type: "none" } : "none";
  }
  return restore;
}

function isFreeModelID(id) {
  return id.endsWith("-free") || KNOWN_FREE.has(id);
}

// ---------- SSE aggregation (free tier forces upstream streaming) ----------
function parseSSE(text) {
  const events = [];
  for (const line of text.split("\n")) {
    const l = line.trim();
    if (!l.startsWith("data:")) continue;
    const payload = l.slice(5).trim();
    if (!payload || payload === "[DONE]") continue;
    try {
      events.push(JSON.parse(payload));
    } catch {}
  }
  return events;
}

function aggregateChatSSE(text, model) {
  const events = parseSSE(text);
  if (!events.length) return null;
  let id = "", created = 0, finish = "", role = "assistant", usage = null;
  const content = [], reasoning = [];
  const tools = new Map(); // index -> {id, name, args}
  for (const e of events) {
    if (e.error) return null;
    if (e.id && !id) id = e.id;
    if (e.created && !created) created = e.created;
    if (e.usage) usage = e.usage;
    const ch = e.choices && e.choices[0];
    if (!ch) continue;
    if (ch.finish_reason) finish = ch.finish_reason;
    const d = ch.delta || {};
    if (d.role) role = d.role;
    if (typeof d.content === "string") content.push(d.content);
    if (typeof d.reasoning_content === "string") reasoning.push(d.reasoning_content);
    if (Array.isArray(d.tool_calls)) {
      for (const tc of d.tool_calls) {
        const idx = typeof tc.index === "number" ? tc.index : 0;
        if (!tools.has(idx)) tools.set(idx, { id: "", name: "", args: "" });
        const acc = tools.get(idx);
        if (tc.id) acc.id = tc.id;
        if (tc.function) {
          if (tc.function.name) acc.name = tc.function.name;
          if (typeof tc.function.arguments === "string") acc.args += tc.function.arguments;
        }
      }
    }
  }
  if (!id && !content.length && !tools.size) return null;
  const message = { role, content: content.join("") };
  if (reasoning.length) message.reasoning_content = reasoning.join("");
  if (tools.size) {
    message.tool_calls = [...tools.entries()].sort((a, b) => a[0] - b[0])
      .map(([, t]) => ({ id: t.id || "call_0", type: "function", function: { name: t.name, arguments: t.args } }));
  }
  return {
    id: id || "chatcmpl-gateway",
    object: "chat.completion",
    created: created || Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, message, finish_reason: finish || "stop" }],
    ...(usage ? { usage } : {}),
  };
}

function aggregateAnthropicSSE(text, model) {
  const events = parseSSE(text);
  if (!events.length) return null;
  let id = "", stop = null, usage = null;
  const blocks = []; // {type, text?} | {type:'tool_use', id, name, _json}
  for (const e of events) {
    if (e.type === "error") return null;
    if (e.type === "message_start" && e.message) {
      id = e.message.id || id;
      if (e.message.usage) usage = { ...e.message.usage };
    }
    if (e.type === "content_block_start" && e.content_block) {
      const cb = e.content_block;
      if (cb.type === "text") blocks[e.index] = { type: "text", text: cb.text || "" };
      else if (cb.type === "tool_use")
        blocks[e.index] = { type: "tool_use", id: cb.id, name: cb.name, _json: "" };
      else if (cb.type === "thinking") blocks[e.index] = { type: "thinking", thinking: cb.thinking || "" };
      else blocks[e.index] = { ...cb };
    }
    if (e.type === "content_block_delta" && e.delta) {
      const b = blocks[e.index] || (blocks[e.index] = {});
      if (e.delta.type === "text_delta") b.text = (b.text || "") + (e.delta.text || "");
      else if (e.delta.type === "input_json_delta") b._json = (b._json || "") + (e.delta.partial_json || "");
      else if (e.delta.type === "thinking_delta") b.thinking = (b.thinking || "") + (e.delta.thinking || "");
    }
    if (e.type === "message_delta") {
      if (e.delta && e.delta.stop_reason) stop = e.delta.stop_reason;
      if (e.usage) usage = { ...(usage || {}), ...e.usage };
    }
  }
  const content = blocks.filter(Boolean).map((b) => {
    if (b.type === "tool_use") {
      let input = {};
      try { input = b._json ? JSON.parse(b._json) : {}; } catch { input = {}; }
      return { type: "tool_use", id: b.id, name: b.name, input };
    }
    const out = { type: b.type };
    if (b.type === "text") out.text = b.text || "";
    if (b.type === "thinking") out.thinking = b.thinking || "";
    return out;
  });
  return {
    id: id || "msg_gateway",
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: stop || "end_turn",
    stop_sequence: null,
    usage: usage || { input_tokens: 0, output_tokens: 0 },
  };
}

function aggregateResponsesSSE(text) {
  // The Responses stream ends with a response.completed event carrying the
  // full response object — just dig it out.
  for (const e of parseSSE(text)) {
    if (e.type === "response.completed" && e.response) return e.response;
    if (e.type === "response.failed" || e.type === "error") return null;
  }
  return null;
}

function restoreNamesDeep(obj, restore) {
  if (!obj || typeof obj !== "object" || !Object.keys(restore).length) return obj;
  if (Array.isArray(obj)) {
    for (const v of obj) restoreNamesDeep(v, restore);
    return obj;
  }
  for (const [k, v] of Object.entries(obj)) {
    if (k === "name" && typeof v === "string" && restore[v]) obj[k] = restore[v];
    else restoreNamesDeep(v, restore);
  }
  return obj;
}

// Streaming passthrough with tool-name restore, line-wise.
function sseRewriteStream(restore) {
  if (!Object.keys(restore).length) return null;
  const dec = new TextDecoder();
  const enc = new TextEncoder();
  let buf = "";
  const rewriteLine = (line) => {
    if (!line.startsWith("data:")) return line;
    const nl = line.endsWith("\n");
    const payload = line.replace(/^data:\s*/, "").trim();
    if (!payload || payload === "[DONE]") return line;
    try {
      const j = JSON.parse(payload);
      const before = JSON.stringify(j);
      restoreNamesDeep(j, restore);
      if (JSON.stringify(j) !== before) return "data: " + JSON.stringify(j) + (nl ? "\n" : "");
    } catch {}
    return line;
  };
  return new TransformStream({
    transform(chunk, ctrl) {
      buf += dec.decode(chunk, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        ctrl.enqueue(enc.encode(rewriteLine(buf.slice(0, i + 1))));
        buf = buf.slice(i + 1);
      }
    },
    flush(ctrl) {
      if (buf) ctrl.enqueue(enc.encode(rewriteLine(buf)));
    },
  });
}

// ---------- key resolution ----------
function resolveKeys(request, url, env) {
  const h = request.headers;
  const auth = h.get("authorization") || "";
  const bearer = auth.toLowerCase().startsWith("bearer ") ? auth.slice(7).trim() : "";
  const gw = (env.GATEWAY_KEY || "").trim();

  let zenKey =
    (h.get("x-zen-key") || "").trim() ||
    (url.searchParams.get("key") || "").trim() ||
    (bearer && !(gw && bearer === gw) ? bearer : "") ||
    (h.get("x-api-key") || "").trim() ||
    "";

  if (gw) {
    const clientKey = (h.get("x-gateway-key") || "").trim() || (bearer === gw ? bearer : "");
    if (!clientKey)
      return { error: json({ error: { type: "auth_error", message: "Missing gateway key. Send 'Authorization: Bearer <GATEWAY_KEY>' or 'X-Gateway-Key'." } }, 401) };
    if (clientKey !== gw)
      return { error: json({ error: { type: "auth_error", message: "Invalid gateway key." } }, 401) };
  }

  if (!zenKey) zenKey = (env.ZEN_API_KEY || "").trim();
  // 'public' or empty -> free tier
  const free = !zenKey || zenKey === "public";
  return { zenKey: free ? null : zenKey, free };
}

// Pass-through for the Jev/SystemOne decision wire (free tier verified).
// Body: {"model":"jev-1.13-free","state":...,"questions":{...}} — see
// https://openrouter.ai/docs/guides/community/jev
async function handleSystemOne(request, env) {
  const uaVersion = (env.OC_UA_VERSION || OC_UA_VERSION).trim();
  const session = ocSessionID();
  const headers = {
    "content-type": "application/json",
    authorization: "Bearer public",
    "user-agent": `opencode/latest/${uaVersion}/cli`,
    "x-opencode-client": "cli",
    "x-opencode-project": "global",
    "x-opencode-session": session,
    "x-session-id": session,
    "x-session-affinity": session,
    "x-opencode-session-id": session,
    accept: "application/json",
  };
  const raw = await request.text();
  let resp;
  try {
    resp = await fetch(UPSTREAM + SYSTEMONE_UPSTREAM, { method: "POST", headers, body: raw });
  } catch (e) {
    return json({ error: { type: "upstream_error", message: `Upstream fetch failed: ${e.message}` } }, 502);
  }
  const text = await resp.text();
  return new Response(text, {
    status: resp.status,
    headers: { "content-type": resp.headers.get("content-type") || "application/json", ...CORS, "x-gateway-mode": "free", "x-gateway-upstream-status": String(resp.status) },
  });
}

async function handleModels(env) {
  const r = await fetch(UPSTREAM + "/zen/v1/models");
  const data = await r.json();
  if (data && Array.isArray(data.data))
    for (const m of data.data) m.free = isFreeModelID(m.id || "");
  return json(data, r.status);
}

async function handleFree(request, url, pathname, env, bodyJson = null, bodyText = null) {
  const wire = pathname === "/v1/messages" ? "messages" : pathname === "/v1/responses" ? "responses" : "chat";
  let body = bodyJson;
  if (!body) {
    try {
      body = bodyText != null ? JSON.parse(bodyText) : await request.json();
    } catch {
      return json({ error: { type: "invalid_request", message: "Body must be JSON." } }, 400);
    }
  }
  if (!body || typeof body !== "object" || !body.model)
    return json({ error: { type: "invalid_request", message: "'model' is required." } }, 400);
  const model = body.model;
  if (!isFreeModelID(model))
    return json(
      {
        error: {
          type: "not_free_model",
          message: `Model '${model}' is not in the free catalog. Free models: see GET /v1/models (free:true). For paid models pass a Zen key (https://opencode.ai/auth) via Authorization/X-Zen-Key.`,
        },
      },
      400
    );

  const clientStream = body.stream === true;

  // Bridge the client wire to the upstream chat wire.
  let chatBody, restore = {};
  if (wire === "messages") chatBody = anthropicToChatBody(body);
  else if (wire === "responses") chatBody = responsesToChatBody(body);
  else { chatBody = body; restore = injectTools(chatBody, "chat"); }

  chatBody.model = model;
  chatBody.stream = true; // upstream gate: stream must be true
  chatBody.stream_options = { ...(chatBody.stream_options || {}), include_usage: true };
  // Always ensure the 4 required tools exist (adds missing stubs; sets
  // tool_choice:none only when the client brought no tools at all).
  restore = injectTools(chatBody, "chat");

  const uaVersion = (env.OC_UA_VERSION || OC_UA_VERSION).trim();
  const session = ocSessionID();
  const headers = {
    "content-type": "application/json",
    authorization: "Bearer public",
    "user-agent": `opencode/latest/${uaVersion}/cli`,
    "x-opencode-client": "cli",
    "x-opencode-project": "global",
    "x-opencode-session": session,
    "x-session-id": session,
    "x-session-affinity": session,
    "x-opencode-session-id": session,
    accept: "text/event-stream",
  };
  if (wire === "messages") {
    // Upstream is the chat wire now; no anthropic headers needed upstream.
  }
  // NOTE: x-opencode-request must NOT be sent on /inference/* paths (403).

  const upstreamUrl = `${UPSTREAM}/inference/openai/v1/chat/completions`;

  let resp;
  try {
    resp = await fetch(upstreamUrl, { method: "POST", headers, body: JSON.stringify(chatBody) });
  } catch (e) {
    return json({ error: { type: "upstream_error", message: `Upstream fetch failed: ${e.message}` } }, 502);
  }

  if (!resp.ok) {
    const text = await resp.text();
    return new Response(text, {
      status: resp.status,
      headers: { "content-type": resp.headers.get("content-type") || "application/json", ...CORS, "x-gateway-mode": "free", "x-gateway-upstream-status": String(resp.status) },
    });
  }

  const raw = await resp.text();

  if (clientStream) {
    // Native chat wire: stream upstream chunks straight through (with
    // tool-name restore). Bridged wires: buffer + translate so the client
    // gets correct Anthropic/Responses SSE events.
    if (wire === "chat") {
      const ts = sseRewriteStream(restore);
      const stream = ts ? resp.body.pipeThrough(ts) : raw;
      return new Response(stream, {
        status: 200,
        headers: { "content-type": "text/event-stream", "cache-control": "no-store", ...CORS, "x-gateway-mode": "free" },
      });
    }
    const agg = aggregateChatSSE(raw, model);
    restoreNamesDeep(agg, restore);
    if (!agg)
      return json({ error: { type: "upstream_error", message: "Could not translate upstream stream.", raw: raw.slice(0, 400) } }, 502, { "x-gateway-mode": "free" });
    const sseBody = wire === "messages" ? sseFromAnthropicMessage(chatToAnthropic(agg)) : sseFromResponsesObject(chatToResponses(agg));
    return new Response(sseBody, {
      status: 200,
      headers: { "content-type": "text/event-stream", "cache-control": "no-store", ...CORS, "x-gateway-mode": "free" },
    });
  }

  // client wanted non-streaming: aggregate
  const agg = aggregateChatSSE(raw, model);
  restoreNamesDeep(agg, restore);
  if (!agg)
    return json(
      { error: { type: "upstream_error", message: "Could not aggregate upstream stream into a non-streaming response.", raw: raw.slice(0, 400) } },
      502,
      { "x-gateway-mode": "free" }
    );
  if (wire === "messages") return json(chatToAnthropic(agg), 200, { "x-gateway-mode": "free" });
  if (wire === "responses") return json(chatToResponses(agg), 200, { "x-gateway-mode": "free" });
  return json(agg, 200, { "x-gateway-mode": "free" });
}

async function handleKeyed(request, url, pathname, zenKey, bodyText = null, bodyJson = null, env = null) {
  const upstreamUrl = UPSTREAM + "/zen" + pathname + url.search;
  const headers = new Headers({ "content-type": "application/json", authorization: `Bearer ${zenKey}` });
  headers.set("x-api-key", zenKey);
  const av = request.headers.get("anthropic-version");
  if (av) headers.set("anthropic-version", av);
  const ab = request.headers.get("anthropic-beta");
  if (ab) headers.set("anthropic-beta", ab);
  const init = { method: request.method, headers };
  if (request.method === "POST") init.body = bodyText != null ? bodyText : await request.text();
  let resp;
  try {
    resp = await fetch(upstreamUrl, init);
  } catch (e) {
    return json({ error: { type: "upstream_error", message: `Upstream fetch failed: ${e.message}` } }, 502);
  }
  // Free-model fallback: harnesses (Hermes, OpenAI SDKs) always send an
  // Authorization header — often a placeholder. If keyed mode fails on a
  // model we can serve for free, retry through the free fingerprint.
  const model = bodyJson?.model;
  const badKey = resp.status === 401 || resp.status === 403 || resp.status >= 500;
  if (badKey && model && isFreeModelID(model) &&
      (pathname === "/v1/chat/completions" || pathname === "/v1/messages" || pathname === "/v1/responses")) {
    return handleFree(request, url, pathname, env, bodyJson, bodyText);
  }
  const out = new Headers();
  const ct = resp.headers.get("content-type");
  if (ct) out.set("content-type", ct);
  out.set("x-gateway-mode", "keyed");
  out.set("x-gateway-upstream-status", String(resp.status));
  out.set("access-control-allow-origin", "*");
  return new Response(resp.body, { status: resp.status, headers: out });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });

    if (path === "/" || path === "/health") {
      return json({
        ok: true,
        service: "opencode-gateway",
        modes: {
          free: "no key / Bearer public -> OpenCode free models (client-shape fingerprint)",
          keyed: "Zen key via Authorization/X-Zen-Key/x-api-key/?key= (or ZEN_API_KEY secret) -> /zen/v1 proxy",
        },
        routes: ["/v1/models", "/v1/chat/completions", "/v1/messages", "/v1/responses"],
        pinned_oc_version: env.OC_UA_VERSION || OC_UA_VERSION,
        gateway_key_required: !!env.GATEWAY_KEY,
      });
    }

    if (path === "/v1/models" && request.method === "GET") return handleModels(env);

    if (request.method === "POST" && path === SYSTEMONE_PATH) {
      return handleSystemOne(request, env);
    }

    if (request.method !== "POST" || !PROXY_PATHS.has(path)) {
      return json(
        {
          error: {
            type: "not_found",
            message: `Unsupported ${request.method} ${path}. Use POST /v1/chat/completions, /v1/messages, /v1/responses, or GET /v1/models.`,
          },
        },
        404
      );
    }

    const resolved = resolveKeys(request, url, env);
    if (resolved.error) return resolved.error;

    // Parse body once so both modes share it (also enables free-model fallback).
    let bodyJson = null, bodyText = null;
    if (request.method === "POST") {
      bodyText = await request.text();
      try { bodyJson = JSON.parse(bodyText); } catch {}
    }

    if (resolved.free) {
      if (path === "/v1/chat/completions" || path === "/v1/messages" || path === "/v1/responses")
        return handleFree(request, url, path, env, bodyJson, bodyText);
      return json({ error: { type: "auth_error", message: "Free mode supports chat/completions, messages and responses only." } }, 401);
    }
    return handleKeyed(request, url, path, resolved.zenKey, bodyText, bodyJson, env);
  },
};
