// bridge.js — wire-protocol bridging for opencode-gateway free mode.
// All free models are served on the OpenAI chat/completions wire, so client
// requests in Anthropic Messages or OpenAI Responses format are converted to
// chat on the way up, and chat responses are translated back on the way down
// (same architecture as 12errh/zen-proxy).

export function safeParse(args) {
  if (typeof args !== "string") return args ?? {};
  try {
    const v = JSON.parse(args);
    return v && typeof v === "object" ? v : {};
  } catch {
    return {};
  }
}

export function parseSSEText(text) {
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

// Aggregate an upstream chat.completion.chunk SSE body into one
// chat.completion object.
export function aggregateChatSSE(text, model) {
  const events = parseSSEText(text);
  if (!events.length) return null;
  let id = "", created = 0, finish = "", role = "assistant", usage = null;
  const content = [], reasoning = [];
  const tools = new Map();
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

// ---- Anthropic Messages request -> chat request ----
export function anthropicToChatBody(body) {
  const out = [];
  const sys = typeof body.system === "string"
    ? body.system
    : Array.isArray(body.system)
      ? body.system.map((b) => b?.text || "").join("\n")
      : "";
  if (sys) out.push({ role: "system", content: sys });
  for (const m of Array.isArray(body.messages) ? body.messages : []) {
    const role = m?.role === "assistant" ? "assistant" : "user";
    if (typeof m?.content === "string") {
      out.push({ role, content: m.content });
      continue;
    }
    const blocks = Array.isArray(m?.content) ? m.content : [];
    const toolCalls = [];
    const parts = [];
    const flush = () => {
      const text = parts.filter((p) => p?.type === "text" && p.text).map((p) => p.text).join("");
      const images = parts.filter((p) => p?.type === "image_url");
      if (text || images.length)
        out.push({ role, content: images.length ? [...(text ? [{ type: "text", text }] : []), ...images] : text });
      parts.length = 0;
    };
    for (const b of blocks) {
      const t = b?.type;
      if (t === "text") parts.push({ type: "text", text: b.text ?? "" });
      else if (t === "image" && b.source?.type === "base64")
        parts.push({ type: "image_url", image_url: { url: `data:${b.source.media_type};base64,${b.source.data}` } });
      else if (t === "tool_use")
        toolCalls.push({
          id: b.id ?? "call_0",
          type: "function",
          function: {
            name: b.name ?? "unknown",
            arguments: typeof b.input === "string" ? b.input : JSON.stringify(b.input ?? {}),
          },
        });
      else if (t === "tool_result") {
        flush();
        out.push({
          role: "tool",
          tool_call_id: b.tool_use_id ?? b.id ?? "call_0",
          content:
            typeof b.content === "string"
              ? b.content
              : Array.isArray(b.content)
                ? b.content.filter((p) => p?.type === "text").map((p) => p.text).join("") || " "
                : " ",
        });
      }
    }
    flush();
    if (toolCalls.length) out.push({ role: "assistant", content: "", tool_calls: toolCalls });
  }
  if (!out.length) out.push({ role: "user", content: "" });

  const chat = { model: body.model, messages: out, stream: body.stream === true };
  const mt = Number(body.max_tokens);
  if (Number.isFinite(mt) && mt > 0) chat.max_tokens = Math.floor(mt);
  if (body.temperature != null) chat.temperature = body.temperature;
  if (body.top_p != null) chat.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length) chat.stop = body.stop_sequences.slice(0, 4);
  const tools = Array.isArray(body.tools)
    ? body.tools
        .filter((t) => t?.name)
        .map((t) => ({
          type: "function",
          function: {
            name: t.name,
            ...(t.description ? { description: t.description } : {}),
            parameters: t.input_schema ?? { type: "object", properties: {} },
          },
        }))
    : [];
  if (tools.length) chat.tools = tools;
  const tc = body.tool_choice;
  if (tc && typeof tc === "object" && tc.type === "tool") chat.tool_choice = { type: "function", function: { name: tc.name } };
  else if (tc && typeof tc === "object" && tc.type === "none") chat.tool_choice = "none";
  else if (tc && typeof tc === "object" && tc.type === "any") chat.tool_choice = "required";
  else if (typeof tc === "string") chat.tool_choice = tc;
  return chat;
}

// ---- OpenAI Responses request -> chat request ----
export function responsesToChatBody(body) {
  const msgs = [];
  if (body.instructions) msgs.push({ role: "system", content: body.instructions });
  const input = body.input;
  if (typeof input === "string") {
    msgs.push({ role: "user", content: input });
  } else if (Array.isArray(input)) {
    for (const item of input) {
      const type = item?.type || (item?.role ? "message" : "");
      if (type === "message" && item.role) {
        const role = item.role === "assistant" ? "assistant" : item.role === "system" || item.role === "developer" ? "system" : "user";
        const content =
          typeof item.content === "string"
            ? item.content
            : Array.isArray(item.content)
              ? item.content.map((p) => (typeof p === "string" ? p : p?.type === "output_text" ? p.text : p?.text || "")).join("")
              : "";
        msgs.push({ role, content });
      } else if (type === "function_call") {
        msgs.push({
          role: "assistant",
          content: "",
          tool_calls: [{
            id: item.call_id || item.id || "call_0",
            type: "function",
            function: {
              name: item.name || "unknown",
              arguments: typeof item.arguments === "string" ? item.arguments : JSON.stringify(item.arguments ?? {}),
            },
          }],
        });
      } else if (type === "function_call_output") {
        msgs.push({
          role: "tool",
          tool_call_id: item.call_id || "call_0",
          content: typeof item.output === "string" ? item.output : JSON.stringify(item.output ?? {}),
        });
      }
    }
  }
  if (!msgs.length) msgs.push({ role: "user", content: "" });

  const chat = { model: body.model, messages: msgs, stream: body.stream === true };
  const mt = Number(body.max_output_tokens);
  if (Number.isFinite(mt) && mt > 0) chat.max_tokens = Math.floor(mt);
  if (body.temperature != null) chat.temperature = body.temperature;
  if (body.top_p != null) chat.top_p = body.top_p;
  const tools = Array.isArray(body.tools)
    ? body.tools
        .filter((t) => t?.name && (t.type === "function" || !t.type))
        .map((t) => ({
          type: "function",
          function: {
            name: t.name,
            ...(t.description ? { description: t.description } : {}),
            parameters: t.parameters ?? { type: "object", properties: {} },
          },
        }))
    : [];
  if (tools.length) chat.tools = tools;
  const tc = body.tool_choice;
  if (tc === "none" || tc === "auto" || tc === "required") chat.tool_choice = tc;
  else if (tc && typeof tc === "object" && tc.type === "function") chat.tool_choice = { type: "function", function: { name: tc.name } };
  return chat;
}

// ---- chat completion -> Anthropic message ----
const STOP_MAP = { stop: "end_turn", length: "max_tokens", tool_calls: "tool_use", function_call: "tool_use", content_filter: "end_turn" };

export function chatToAnthropic(agg) {
  const m = agg?.choices?.[0]?.message ?? {};
  const blocks = [];
  if (m.reasoning_content) blocks.push({ type: "thinking", thinking: m.reasoning_content });
  const text = typeof m.content === "string" ? m.content : "";
  if (text) blocks.push({ type: "text", text });
  for (const tc of m.tool_calls ?? [])
    blocks.push({ type: "tool_use", id: tc.id || "toolu_0", name: tc.function?.name || "unknown", input: safeParse(tc.function?.arguments) });
  const u = agg.usage || {};
  return {
    id: "msg_" + String(agg.id || "gw").replace(/[^a-zA-Z0-9]/g, "").slice(0, 40),
    type: "message",
    role: "assistant",
    model: agg.model,
    content: blocks.length ? blocks : [{ type: "text", text: "" }],
    stop_reason: STOP_MAP[agg.choices?.[0]?.finish_reason] || "end_turn",
    stop_sequence: null,
    usage: { input_tokens: u.prompt_tokens ?? 0, output_tokens: u.completion_tokens ?? 0 },
  };
}

// ---- chat completion -> Responses object ----
export function chatToResponses(agg) {
  const m = agg?.choices?.[0]?.message ?? {};
  const text = typeof m.content === "string" ? m.content : "";
  const output = [{
    id: "msg_0",
    type: "message",
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text, annotations: [] }],
  }];
  for (const tc of m.tool_calls ?? [])
    output.push({
      id: "fc_" + (output.length - 1),
      type: "function_call",
      name: tc.function?.name || "unknown",
      call_id: tc.id || "call_0",
      arguments: tc.function?.arguments || "{}",
      status: "completed",
    });
  const u = agg.usage || {};
  return {
    id: "resp_" + String(agg.id || "gw").replace(/[^a-zA-Z0-9]/g, "").slice(0, 40),
    object: "response",
    created_at: agg.created || Math.floor(Date.now() / 1000),
    status: "completed",
    model: agg.model,
    output,
    usage: {
      input_tokens: u.prompt_tokens ?? 0,
      output_tokens: u.completion_tokens ?? 0,
      total_tokens: u.total_tokens ?? 0,
    },
  };
}

// ---- SSE emission from a final object (buffered translation) ----
const sse = (event, data) => (event ? `event: ${event}\n` : "") + `data: ${JSON.stringify(data)}\n\n`;

export function sseFromAnthropicMessage(msg) {
  let out = sse("message_start", {
    type: "message_start",
    message: {
      id: msg.id, type: "message", role: "assistant", model: msg.model,
      content: [], stop_reason: null, stop_sequence: null,
      usage: { input_tokens: msg.usage.input_tokens, output_tokens: 0 },
    },
  });
  msg.content.forEach((b, i) => {
    const block = b.type === "tool_use" ? { type: "tool_use", id: b.id, name: b.name, input: {} } : { type: b.type, ...(b.type === "thinking" ? { thinking: "" } : { text: "" }) };
    out += sse("content_block_start", { type: "content_block_start", index: i, content_block: block });
    if (b.type === "text" && b.text) out += sse("content_block_delta", { type: "content_block_delta", index: i, delta: { type: "text_delta", text: b.text } });
    if (b.type === "thinking" && b.thinking) out += sse("content_block_delta", { type: "content_block_delta", index: i, delta: { type: "thinking_delta", thinking: b.thinking } });
    if (b.type === "tool_use") out += sse("content_block_delta", { type: "content_block_delta", index: i, delta: { type: "input_json_delta", partial_json: JSON.stringify(b.input) } });
    out += sse("content_block_stop", { type: "content_block_stop", index: i });
  });
  out += sse("message_delta", { type: "message_delta", delta: { stop_reason: msg.stop_reason, stop_sequence: null }, usage: { output_tokens: msg.usage.output_tokens } });
  out += sse("message_stop", { type: "message_stop" });
  return out;
}

export function sseFromResponsesObject(resp) {
  let out = sse("", { type: "response.created", response: resp });
  const msg = resp.output.find((o) => o.type === "message");
  const text = msg?.content?.find((c) => c.type === "output_text")?.text || "";
  if (text) out += sse("", { type: "response.output_text.delta", item_id: "msg_0", output_index: 0, content_index: 0, delta: text });
  out += sse("", { type: "response.completed", response: resp });
  return out;
}
