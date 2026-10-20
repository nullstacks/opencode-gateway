# bridge.py â€” wire-protocol bridging for opencode-gateway free mode.
# Python port of bridge.js. All free models are served on the OpenAI
# chat/completions wire, so client requests in Anthropic Messages or OpenAI
# Responses format are converted to chat on the way up, and chat responses
# are translated back on the way down (same architecture as 12errh/zen-proxy).

import json
import time


def _dumps(value):
    """JSON.stringify-compatible output: compact separators, real UTF-8."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def safe_parse(args):
    if not isinstance(args, str):
        return args if args is not None else {}
    try:
        v = json.loads(args)
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def parse_sse_text(text):
    """Extract data: payloads from an SSE body as parsed JSON objects."""
    events = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except (json.JSONDecodeError, ValueError):
            pass
    return events


def aggregate_chat_sse(text, model):
    """Aggregate an upstream chat.completion.chunk SSE body into one
    chat.completion object. Returns None when the stream carries no usable
    content (or an error event)."""
    events = parse_sse_text(text)
    if not events:
        return None
    msg_id, created, finish, role, usage = "", 0, "", "assistant", None
    content_parts, reasoning_parts = [], []
    tools = {}  # index -> {id, name, args}
    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get("error"):
            return None
        if e.get("id") and not msg_id:
            msg_id = e["id"]
        if e.get("created") and not created:
            created = e["created"]
        if e.get("usage"):
            usage = e["usage"]
        choices = e.get("choices")
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
        delta = choice.get("delta") or {}
        if delta.get("role"):
            role = delta["role"]
        if isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])
        if isinstance(delta.get("reasoning_content"), str):
            reasoning_parts.append(delta["reasoning_content"])
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index") if isinstance(tc.get("index"), int) else 0
            acc = tools.setdefault(idx, {"id": "", "name": "", "args": ""})
            if tc.get("id"):
                acc["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                acc["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                acc["args"] += fn["arguments"]
    if not msg_id and not content_parts and not tools:
        return None
    message = {"role": role, "content": "".join(content_parts)}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tools:
        message["tool_calls"] = [
            {"id": t["id"] or "call_0", "type": "function",
             "function": {"name": t["name"], "arguments": t["args"]}}
            for _, t in sorted(tools.items())
        ]
    import time
    completion = {
        "id": msg_id or "chatcmpl-gateway",
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish or "stop"}],
    }
    if usage:
        completion["usage"] = usage
    return completion


def aggregate_anthropic_sse(text, model):
    """Aggregate an upstream Anthropic Messages SSE stream into one message
    object (used by the keyed-mode Anthropic passthrough in bridge-less mode)."""
    events = parse_sse_text(text)
    if not events:
        return None
    msg_id, stop, usage = "", None, None
    blocks = []  # dict by index
    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get("type") == "error":
            return None
        if e.get("type") == "message_start" and isinstance(e.get("message"), dict):
            msg_id = e["message"].get("id") or msg_id
            if e["message"].get("usage"):
                usage = dict(e["message"]["usage"])
        if e.get("type") == "content_block_start" and isinstance(e.get("content_block"), dict):
            cb = e["content_block"]
            idx = e.get("index", len(blocks))
            if cb.get("type") == "text":
                blocks.insert(idx, {"type": "text", "text": cb.get("text") or ""})
            elif cb.get("type") == "tool_use":
                blocks.insert(idx, {"type": "tool_use", "id": cb.get("id"), "name": cb.get("name"), "_json": ""})
            elif cb.get("type") == "thinking":
                blocks.insert(idx, {"type": "thinking", "thinking": cb.get("thinking") or ""})
            else:
                blocks.insert(idx, dict(cb))
        if e.get("type") == "content_block_delta" and isinstance(e.get("delta"), dict):
            idx = e.get("index", 0)
            while len(blocks) <= idx:
                blocks.append({})
            b = blocks[idx]
            d = e["delta"]
            if d.get("type") == "text_delta":
                b["text"] = (b.get("text") or "") + (d.get("text") or "")
            elif d.get("type") == "input_json_delta":
                b["_json"] = (b.get("_json") or "") + (d.get("partial_json") or "")
            elif d.get("type") == "thinking_delta":
                b["thinking"] = (b.get("thinking") or "") + (d.get("thinking") or "")
        if e.get("type") == "message_delta":
            if isinstance(e.get("delta"), dict) and e["delta"].get("stop_reason"):
                stop = e["delta"]["stop_reason"]
            if e.get("usage"):
                usage = {**(usage or {}), **e["usage"]}
    content = []
    for b in [b for b in blocks if b]:
        if b.get("type") == "tool_use":
            try:
                inp = json.loads(b.get("_json")) if b.get("_json") else {}
            except (json.JSONDecodeError, ValueError):
                inp = {}
            content.append({"type": "tool_use", "id": b.get("id"), "name": b.get("name"), "input": inp})
        else:
            out = {"type": b.get("type")}
            if b.get("type") == "text":
                out["text"] = b.get("text") or ""
            if b.get("type") == "thinking":
                out["thinking"] = b.get("thinking") or ""
            content.append(out)
    return {
        "id": msg_id or "msg_gateway",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop or "end_turn",
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 0, "output_tokens": 0},
    }


def aggregate_responses_sse(text):
    """The Responses stream ends with a response.completed event carrying the
    full response object â€” just dig it out."""
    for e in parse_sse_text(text):
        if e.get("type") == "response.completed" and e.get("response"):
            return e["response"]
        if e.get("type") in ("response.failed", "error"):
            return None
    return None


# ---- Anthropic Messages request -> chat request ----

def anthropic_to_chat_body(body):
    out = []
    system = body.get("system")
    if isinstance(system, str):
        sys_text = system
    elif isinstance(system, list):
        sys_text = "\n".join((b or {}).get("text") or "" for b in system)
    else:
        sys_text = ""
    if sys_text:
        out.append({"role": "system", "content": sys_text})

    for m in body.get("messages") or []:
        role = "assistant" if m.get("role") == "assistant" else "user"
        if isinstance(m.get("content"), str):
            out.append({"role": role, "content": m["content"]})
            continue
        blocks = m.get("content") if isinstance(m.get("content"), list) else []
        tool_calls = []
        parts = []

        def flush():
            text = "".join(p["text"] for p in parts if p.get("type") == "text" and p.get("text"))
            images = [p for p in parts if p.get("type") == "image_url"]
            if text or images:
                if images:
                    content_items = ([{"type": "text", "text": text}] if text else []) + images
                    out.append({"role": role, "content": content_items})
                else:
                    out.append({"role": role, "content": text})
            parts.clear()

        for b in blocks:
            btype = (b or {}).get("type")
            if btype == "text":
                parts.append({"type": "text", "text": b.get("text") or ""})
            elif btype == "image" and (b.get("source") or {}).get("type") == "base64":
                src = b["source"]
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:{src.get('media_type')};base64,{src.get('data')}"}})
            elif btype == "tool_use":
                tool_calls.append({
                    "id": b.get("id") or "call_0",
                    "type": "function",
                    "function": {
                        "name": b.get("name") or "unknown",
                        "arguments": b["input"] if isinstance(b.get("input"), str) else _dumps(b.get("input") or {}),
                    },
                })
            elif btype == "tool_result":
                flush()
                content_val = b.get("content")
                if isinstance(content_val, str):
                    content_text = content_val
                elif isinstance(content_val, list):
                    content_text = "".join(p.get("text") for p in content_val
                                           if isinstance(p, dict) and p.get("type") == "text") or " "
                else:
                    content_text = " "
                out.append({"role": "tool",
                            "tool_call_id": b.get("tool_use_id") or b.get("id") or "call_0",
                            "content": content_text})
        flush()
        if tool_calls:
            out.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
    if not out:
        out.append({"role": "user", "content": ""})

    chat = {"model": body.get("model"), "messages": out, "stream": body.get("stream") is True}
    try:
        mt = float(body.get("max_tokens") or 0)
    except (TypeError, ValueError):
        mt = 0
    if mt > 0:
        chat["max_tokens"] = int(mt)
    if body.get("temperature") is not None:
        chat["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        chat["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        chat["stop"] = body["stop_sequences"][:4]
    tools = []
    for t in body.get("tools") or []:
        if isinstance(t, dict) and t.get("name"):
            fn = {"name": t["name"]}
            if t.get("description"):
                fn["description"] = t["description"]
            fn["parameters"] = t.get("input_schema") or {"type": "object", "properties": {}}
            tools.append({"type": "function", "function": fn})
    if tools:
        chat["tools"] = tools
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        if tc.get("type") == "tool":
            chat["tool_choice"] = {"type": "function", "function": {"name": tc.get("name")}}
        elif tc.get("type") == "none":
            chat["tool_choice"] = "none"
        elif tc.get("type") == "any":
            chat["tool_choice"] = "required"
    elif isinstance(tc, str):
        chat["tool_choice"] = tc
    return chat


# ---- OpenAI Responses request -> chat request ----

def responses_to_chat_body(body):
    msgs = []
    if body.get("instructions"):
        msgs.append({"role": "system", "content": body["instructions"]})
    inp = body.get("input")
    if isinstance(inp, str):
        msgs.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            itype = (item or {}).get("type") or ("message" if (item or {}).get("role") else "")
            if itype == "message" and item.get("role"):
                role = item["role"]
                if role == "assistant":
                    role = "assistant"
                elif role in ("system", "developer"):
                    role = "system"
                else:
                    role = "user"
                content = item.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "".join(
                        p if isinstance(p, str)
                        else (p.get("text") or "" if isinstance(p, dict) else "")
                        for p in content
                    )
                else:
                    text = ""
                msgs.append({"role": role, "content": text})
            elif itype == "function_call":
                args = item.get("arguments")
                msgs.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": item.get("call_id") or item.get("id") or "call_0",
                        "type": "function",
                        "function": {
                            "name": item.get("name") or "unknown",
                            "arguments": args if isinstance(args, str) else _dumps(args or {}),
                        },
                    }],
                })
            elif itype == "function_call_output":
                output = item.get("output")
                msgs.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or "call_0",
                    "content": output if isinstance(output, str) else _dumps(output or {}),
                })
    if not msgs:
        msgs.append({"role": "user", "content": ""})

    chat = {"model": body.get("model"), "messages": msgs, "stream": body.get("stream") is True}
    try:
        mt = float(body.get("max_output_tokens") or 0)
    except (TypeError, ValueError):
        mt = 0
    if mt > 0:
        chat["max_tokens"] = int(mt)
    if body.get("temperature") is not None:
        chat["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        chat["top_p"] = body["top_p"]
    tools = []
    for t in body.get("tools") or []:
        if isinstance(t, dict) and t.get("name") and (t.get("type") == "function" or not t.get("type")):
            fn = {"name": t["name"]}
            if t.get("description"):
                fn["description"] = t["description"]
            fn["parameters"] = t.get("parameters") or {"type": "object", "properties": {}}
            tools.append({"type": "function", "function": fn})
    if tools:
        chat["tools"] = tools
    tc = body.get("tool_choice")
    if tc in ("none", "auto", "required"):
        chat["tool_choice"] = tc
    elif isinstance(tc, dict) and tc.get("type") == "function":
        chat["tool_choice"] = {"type": "function", "function": {"name": tc.get("name")}}
    return chat


# ---- chat completion -> Anthropic message ----

_STOP_MAP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
             "function_call": "tool_use", "content_filter": "end_turn"}


def chat_to_anthropic(agg):
    choice = ((agg or {}).get("choices") or [{}])[0]
    m = choice.get("message") or {}
    blocks = []
    if m.get("reasoning_content"):
        blocks.append({"type": "thinking", "thinking": m["reasoning_content"]})
    text = m.get("content") if isinstance(m.get("content"), str) else ""
    if text:
        blocks.append({"type": "text", "text": text})
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        blocks.append({"type": "tool_use", "id": tc.get("id") or "toolu_0",
                       "name": fn.get("name") or "unknown",
                       "input": safe_parse(fn.get("arguments"))})
    u = agg.get("usage") or {} if agg else {}
    clean = "".join(ch for ch in str((agg or {}).get("id") or "gw") if ch.isalnum())[:40]
    return {
        "id": "msg_" + clean,
        "type": "message",
        "role": "assistant",
        "model": (agg or {}).get("model"),
        "content": blocks or [{"type": "text", "text": ""}],
        "stop_reason": _STOP_MAP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": u.get("prompt_tokens") or 0,
                  "output_tokens": u.get("completion_tokens") or 0},
    }


# ---- chat completion -> Responses object ----

def chat_to_responses(agg):
    choice = ((agg or {}).get("choices") or [{}])[0]
    m = choice.get("message") or {}
    text = m.get("content") if isinstance(m.get("content"), str) else ""
    output = [{
        "id": "msg_0",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }]
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        output.append({
            "id": f"fc_{len(output) - 1}",
            "type": "function_call",
            "name": fn.get("name") or "unknown",
            "call_id": tc.get("id") or "call_0",
            "arguments": fn.get("arguments") or "{}",
            "status": "completed",
        })
    u = agg.get("usage") or {} if agg else {}
    clean = "".join(ch for ch in str((agg or {}).get("id") or "gw") if ch.isalnum())[:40]
    import time
    return {
        "id": "resp_" + clean,
        "object": "response",
        "created_at": (agg or {}).get("created") or int(time.time()),
        "status": "completed",
        "model": (agg or {}).get("model"),
        "output": output,
        "usage": {
            "input_tokens": u.get("prompt_tokens") or 0,
            "output_tokens": u.get("completion_tokens") or 0,
            "total_tokens": u.get("total_tokens") or 0,
        },
    }


# ---- SSE emission from a final object (buffered translation) ----

def _sse(event, data):
    return (f"event: {event}\n" if event else "") + f"data: {_dumps(data)}\n\n"


def sse_from_anthropic_message(msg):
    out = _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg["id"], "type": "message", "role": "assistant", "model": msg["model"],
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": msg["usage"]["input_tokens"], "output_tokens": 0},
        },
    })
    for i, b in enumerate(msg["content"]):
        if b["type"] == "tool_use":
            block = {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}
        elif b["type"] == "thinking":
            block = {"type": "thinking", "thinking": ""}
        else:
            block = {"type": b["type"], "text": ""}
        out += _sse("content_block_start", {"type": "content_block_start", "index": i, "content_block": block})
        if b["type"] == "text" and b.get("text"):
            out += _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                "delta": {"type": "text_delta", "text": b["text"]}})
        if b["type"] == "thinking" and b.get("thinking"):
            out += _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                "delta": {"type": "thinking_delta", "thinking": b["thinking"]}})
        if b["type"] == "tool_use":
            out += _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                "delta": {"type": "input_json_delta",
                                                          "partial_json": _dumps(b["input"])}})
        out += _sse("content_block_stop", {"type": "content_block_stop", "index": i})
    out += _sse("message_delta", {"type": "message_delta",
                                  "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
                                  "usage": {"output_tokens": msg["usage"]["output_tokens"]}})
    out += _sse("message_stop", {"type": "message_stop"})
    return out


def sse_from_responses_object(resp):
    out = _sse("", {"type": "response.created", "response": resp})
    message = next((o for o in resp["output"] if o.get("type") == "message"), None)
    text = ""
    if message:
        text = next((c.get("text") or "" for c in message.get("content") or []
                     if c.get("type") == "output_text"), "")
    if text:
        out += _sse("", {"type": "response.output_text.delta", "item_id": "msg_0",
                         "output_index": 0, "content_index": 0, "delta": text})
    out += _sse("", {"type": "response.completed", "response": resp})
    return out
