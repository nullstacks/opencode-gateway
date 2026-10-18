# opencode-gateway

Cloudflare Worker that lets **any** harness (Claude Code, Codex CLI, Aider,
OpenHands, LangChain, curl, ...) use **OpenCode models** through one
OpenAI/Anthropic-compatible endpoint — **no OpenCode account or API key**
needed for the free models.

[![Deploy to Cloudflare Workers](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/nullstacks/opencode-gateway)

## Free models

Verified live against the upstream catalog. `GET /v1/models` returns the
current list with `free: true` flags (additions/removals upstream are picked
up automatically).

| Model | Status (2026-10-18) |
|---|---|
| `big-pickle` | ✅ verified |
| `beam-free` | ✅ verified |
| `deepseek-v4.1-flash-free` | ✅ verified |
| `glm-5.3-flash-free` | ✅ verified |
| `longcat-2.5-preview-free` | ✅ verified |
| `mimo-v2.6-flash-free` | ✅ verified |
| `nemotron-3.5-lightning-free` | ✅ verified |
| `nemotron-3.5-super-vl-free` | ✅ verified |
| `space-bunny-free` | ✅ verified |
| `jev-1.13-free` | ✅ works — but it is a TypeSafe *decision* model, not chat; use `POST /v1/systemone` (below) |

Free-tier catalog is limited-time promos upstream — models can rotate; the
gateway's `/v1/models` always reflects what is currently available, and any
model ending in `-free` (plus `big-pickle`) is routed through the free
fingerprint automatically.

## Endpoints

```
GET  /v1/models            catalog
POST /v1/chat/completions  OpenAI Chat wire (stream + non-stream)
POST /v1/messages          Anthropic Messages wire (Claude Code compatible)
POST /v1/responses         OpenAI Responses wire
GET  /health
```

## Jev / SystemOne (decision models)

`jev-1.13` / `jev-1.13-free` are **not chat models** — they are TypeSafe
decision models (classify state into answers with probabilities/confidence).
They speak their own wire format on a dedicated endpoint, which the gateway
passes through:

```
POST /v1/systemone
{
  "model": "jev-1.13-free",
  "state": "My Stripe integration keeps failing for 3 days.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this",
                   "criteria": {"billing": "Payment issues", "technical": "Bugs"}},
    "is_urgent": {"type": "noul", "instructions": "The customer explicitly requests a refund"}
  }
}
```

Response: `{"answers": {"department": {"choice": "billing", "confidence": 0.44, ...}, ...}}`
Question types: `choice` (≤255 options), `score` (2–10 ordered levels), `noul`
(true/false). Free tier verified working — see
[the Jev guide](https://openrouter.ai/docs/guides/community/jev) and the
[API reference](https://typesafe-jev.com/en/guides/api/) for the full schema.

## How the free tier works

OpenCode's free tier is not account-gated — it is **client-shape gated**.
This worker reproduces the official opencode CLI fingerprint on every free
request (reverse-engineered by the
[opencode2api](https://github.com/6Kmfi6HP/opencode2api) project, documented
in their [UPSTREAM-COMPAT.md](https://github.com/6Kmfi6HP/opencode2api/blob/main/docs/UPSTREAM-COMPAT.md)):

1. Upstream path `/inference/openai/v1/chat/completions` — all free models
   accept the OpenAI chat wire; Anthropic/Responses client requests are
   bridged locally (`bridge.js`, same architecture as
   [zen-proxy](https://github.com/12errh/zen-proxy))
2. `User-Agent: opencode/latest/<ver>/cli` — version pinned (`2.0.29`); the
   gate rejects too-old AND too-new versions
3. `x-opencode-session: ses_` + descending 48-bit timestamp hex + 14 base62
   chars (exact ID encoding from `packages/opencode/src/id/id.ts`)
4. Four session headers; `x-opencode-request` must NOT be sent (403)
5. `stream: true` forced upstream — non-stream clients get local SSE→JSON
   aggregation
6. `tools` must include `bash`, `glob`, `grep`, `read` with non-empty
   `parameters.properties` — stubs are injected automatically, and
   `tool_choice: none` is set when the client brought no tools

> ⚠️ **Fragility:** this is an adversarial compat layer, not a stable API.
> OpenCode has tightened the gate five times between 2026-09-17 and
> 2026-10-15. If free models start returning `403 FreeTierError`, check the
> compat doc above and re-pin the version via the `OC_UA_VERSION` secret.

## Deploy

**Option A — button above** (Deploys to Cloudflare from this repo).

**Option B — CLI:**

```bash
git clone https://github.com/nullstacks/opencode-gateway
cd opencode-gateway
npx wrangler login
npx wrangler deploy
```

### Secrets (optional)

| Secret | Purpose |
|---|---|
| `ZEN_API_KEY` | fallback Zen key — unlocks all 85 paid models for keyless callers |
| `GATEWAY_KEY` | require callers to present this key (lock public deployments) |
| `OC_UA_VERSION` | override the pinned opencode CLI version |

```bash
npx wrangler secret put ZEN_API_KEY
```

## Usage

```bash
# OpenAI client — free model, no key
curl https://<your-worker>.workers.dev/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.3-flash-free","messages":[{"role":"user","content":"hi"}]}'

# Paid model — pass your Zen key from https://opencode.ai/auth
curl ... -H "Authorization: Bearer <zen-key>" -d '{"model":"gpt-5.5",...}'

# Claude Code
export ANTHROPIC_BASE_URL=https://<your-worker>.workers.dev
export ANTHROPIC_API_KEY=free
export ANTHROPIC_MODEL=glm-5.3-flash-free
```

Key resolution order: `X-Zen-Key` header → `?key=` → `Authorization: Bearer`
→ `x-api-key` → `ZEN_API_KEY` secret. `public` / empty = free tier.

## License

MIT
