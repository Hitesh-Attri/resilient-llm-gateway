# resilient-llm-gateway (Module 1)

A from-scratch, provider-agnostic gateway that fronts OpenAI, Anthropic, AWS
Bedrock, and any OpenAI-compatible endpoint (Groq, Gemini, Ollama) behind one
interface, with per-target retries and an ordered fallback chain. This is the
foundation every later module (RAG, agent, MCP server) makes its model calls
through.

## The idea in one line

Callers speak one normalized vocabulary (`ChatRequest` / `ChatResponse`).
Adapters translate to each vendor. The gateway retries a target on transient
errors, then falls over to the next target - failing fast only on requests no
provider could serve.

## Layout

```
resilient-llm-gateway/
├── src/                          # import root (see "Imports" below)
│   ├── core/
│   │   ├── types.py              # normalized vocabulary + StreamChunk + enums
│   │   ├── provider.py           # Provider Protocol, ProviderError, classifiers
│   │   │                         #   (should_failover / is_transient)
│   │   ├── retry.py              # RetryPolicy + full-jitter backoff (pure)
│   │   ├── structured.py         # JSON-schema validation for structured output
│   │   ├── gateway.py            # complete() + stream(); retry + fallback + validate
│   │   ├── config.py             # config-driven chain + retry policy from env
│   │   ├── log.py                # get_logger(); logging setup lives here
│   │   └── log_context.py        # request-id ContextVar (get/set across async)
│   ├── providers/                # one class per provider; SDKs lazy-imported
│   │   ├── openai_compatible.py  # base: shared OpenAI wire protocol
│   │   ├── openai_provider.py    # }
│   │   ├── groq_provider.py      # } thin subclasses: identity + capabilities +
│   │   ├── gemini_provider.py    # } per-provider quirks (reasoning_effort, etc.)
│   │   ├── ollama_provider.py    # }
│   │   ├── anthropic_provider.py # forced tool-use for structured output
│   │   └── bedrock_provider.py   # forced tool-use + sync->async stream bridge
│   ├── api/
│   │   └── routes.py             # POST /v1/chat (buffered), /v2/chat (SSE), GET /health
│   └── main.py                   # FastAPI app + `python src/main.py` entrypoint
├── tests/                        # no API keys needed (fake providers, injected clock)
│   ├── test_gateway_fallback.py  # outer loop: fail-over vs fail-fast
│   ├── test_error_classification.py  # should_failover / is_transient
│   ├── test_retry.py             # inner loop: backoff schedule + retry-then-failover
│   ├── test_streaming.py         # commit boundary: failover before token, not after
│   ├── test_providers.py         # provider identity/capabilities + config wiring
│   ├── test_finish_reason.py     # stop-reason normalization across providers
│   ├── test_reasoning_effort.py  # reasoning_effort resolution + capability gating
│   ├── test_request_validation.py    # ChatRequest bounds/defaults contract
│   └── test_structured_output.py # schema validation + failover on violation
├── infra/                        # OpenTofu + Terragrunt (deployment slice, later)
│   └── README.md
├── .github/workflows/ci.yml      # lint + test on push/PR
├── Dockerfile
├── docker-compose.yml            # gateway + redis (redis staged for cache slice)
├── .dockerignore
├── .gitignore
├── .env.example                  # copy to .env; documents every setting
├── pyproject.toml                # deps + makes src/ the import root + tool config
├── requirements.txt              # runtime deps (for Docker layer caching)
└── requirements-dev.txt          # runtime + pytest / ruff
```

## Imports

Absolute, rooted at `src/`: `from core.gateway import LLMGateway`,
`from providers.openai_provider import OpenAIProvider`. No `app.` prefix, no
relative `..` hops. `src/` is put on the path three ways depending on context:

- **Dev:** `pip install -e .` (reads `package-dir = {"" = "src"}` in pyproject).
- **Tests:** `pythonpath = ["src"]` in pyproject, so no install needed.
- **Docker / direct run:** `PYTHONPATH=/app/src`, and running `python src/main.py`
  adds `src/` automatically (Python puts the script's own dir on the path).

## Dependencies

`pyproject.toml` is the source of truth (loose ranges: "what this app is
compatible with"). `requirements.txt` / `requirements-dev.txt` are *generated*
pinned lockfiles for reproducible Docker builds - don't hand-edit them, regenerate.

Add or change a dependency: edit `pyproject.toml`, then regenerate the lockfiles.
Two toolchains:

```bash
# uv (recommended in 2026 - fast; uv.lock is the real lockfile, commit it)
uv add httpx                     # edits pyproject + updates uv.lock
uv export --no-hashes --no-emit-project -o requirements.txt
uv export --extra dev --no-hashes --no-emit-project -o requirements-dev.txt

# or pip-tools (pip-native)
pip install pip-tools
pip-compile pyproject.toml -o requirements.txt
pip-compile --extra dev pyproject.toml -o requirements-dev.txt
```

Then commit `pyproject.toml` and both regenerated files together.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # editable install + dev tools

cp .env.example .env             # set keys + confirm current model IDs

python src/main.py               # direct run (reload on)
# or: uvicorn main:app --reload  (needs PYTHONPATH=src or the editable install)
```

Docker:

```bash
docker compose up --build        # gateway on :8000, redis on :6379
```

Call it:

```bash
curl -s localhost:8000/v1/chat -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"Explain a token bucket in one sentence."}],
  "system":"You are terse."
}' | jq
```

## Understanding the request body

The model is stateless and, under the hood, sees the whole request flattened
into one labeled transcript that it continues from. Two fields shape that:

- **`messages`** - the conversation, each entry tagged with a **`role`**:
  - `user` - what the human said
  - `assistant` - what the model said on a previous turn
  - `system` - not part of the dialogue; instructions on *how* to behave

  The roles let the model track whose turn it is. The model has no memory between
  calls: to continue a conversation you resend the prior `user`/`assistant` turns,
  and the labels are what let it reconstruct who said what.

- **`system`** - a privileged instruction channel set by you (the app), separate
  from the conversation. It steers *how* the model answers ("be terse", "reply in
  JSON", "you are a support agent for Acme") independent of *what* is asked. This
  is why `ChatRequest` keeps it as its own field: Anthropic/Bedrock take it
  top-level, OpenAI folds it in as a `role: "system"` message, and the adapters
  normalize that difference.

Same question, only `system` changing - watch the *shape* of the answer change
while the facts stay the same:

```bash
# Terse -> one clipped line
curl -s localhost:8000/v1/chat -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"What is a token bucket?"}],
  "system":"You are terse. Answer in one sentence and nothing more."
}' | jq -r .content

# Pedagogical -> longer, structured explanation
curl -s localhost:8000/v1/chat -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"What is a token bucket?"}],
  "system":"You are a distinguished professor. Be thorough and pedagogical."
}' | jq -r .content

# Structured -> machine-parseable output you could consume downstream
curl -s localhost:8000/v1/chat -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"What is a token bucket?"}],
  "system":"Respond only in valid JSON with keys \"definition\" and \"use_case\"."
}' | jq -r .content
```

Steering behavior through the system prompt is most of what prompt engineering
is, and it's where later modules put "answer only from the provided context,
cite sources" (RAG) and "you may call these tools" (agents).

## Test

```bash
pytest                           # fallback + classification tests, no API keys needed
```

## Design decisions worth knowing

- **System prompt is a separate field**, not a message. Anthropic/Bedrock want it
  top-level; OpenAI wants it as a `role: system` message. Keeping it separate lets
  each adapter place it correctly.
- **Fail-over vs fail-fast** is decided by `ProviderError.retryable`. Transient
  (429 / 5xx / timeout) fails over to the next target. A malformed or unauthorized
  request (4xx) fails fast - another provider would reject it too.
- **Lazy SDK imports.** The core imports with zero vendor SDKs installed; adapters
  import their SDK only when constructed.

## Reliability: two nested loops

- **Inner (retry):** on a *transient* failure (real rate limit, 5xx, timeout,
  connection drop) the gateway retries the **same** target up to
  `RETRY_MAX_ATTEMPTS` times, with exponential backoff + full jitter
  (`RETRY_BASE_DELAY` growing to `RETRY_MAX_DELAY`).
- **Outer (fallback):** when a target is exhausted, or fails with a retryable but
  *non-transient* error (no credits, bad key, retired model), the gateway falls
  **over** to the next target.

The `ProviderError.transient` flag drives the inner loop, `retryable` drives the
outer one. A no-credits 429 is retryable-but-not-transient, so it fails over
immediately instead of wasting backoff on a provider that can't recover.

## Streaming (`POST /v2/chat`)

Same request body as `/v1/chat`, but the response is a Server-Sent Events stream
so tokens render live instead of waiting for the whole completion:

```bash
curl -N -s localhost:8000/v2/chat -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"Explain a token bucket."}],
  "system":"You are terse."
}'
```

Events (`-N` disables curl buffering so you see them arrive):

```
data: {"type":"delta","content":"A token "}
data: {"type":"delta","content":"bucket "}
...
data: {"type":"done","provider":"groq","model":"...","usage":{...},"latency_ms":812.4}
```

An abnormal end (e.g. all providers down, or a mid-stream drop) arrives as a
final `{"type":"error","message":"..."}` event.

**The commit boundary.** Retry and fallback can only happen *before the first
token reaches the client* - once a delta is sent, the client has partial output
and the stream is committed to that provider. So all recovery is concentrated in
`gateway._open_stream` (get the first chunk in hand, retrying/failing over as
needed); after that the gateway only relays, and a mid-stream failure surfaces as
an `error` event rather than silently switching providers. `test_streaming.py`
pins both halves of this.

Usage during streaming is provider-dependent: OpenAI and Groq report it via a
final usage chunk (`stream_usage=True`), Anthropic via the final message, Bedrock
via a metadata event; Gemini's compat layer omits it, so `usage` may be zero
there.

## Structured output

Send a JSON Schema as `response_schema` and the gateway makes the model return
conforming JSON, validates it, and hands you a parsed object:

```bash
curl -s localhost:8000/v1/chat -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"Classify the sentiment of: I love this."}],
  "response_schema":{
    "type":"object",
    "properties":{"label":{"type":"string"},"score":{"type":"number"}},
    "required":["label","score"],
    "additionalProperties":false
  }
}' | jq '{content, parsed}'
```

`content` holds the raw JSON string; `parsed` is the validated object.

Schemas nest - use `$defs` for reused shapes, `enum` to constrain values, and
`array` for lists. This extracts structured data from a support ticket:

```bash
curl -s localhost:8000/v1/chat -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"Extract details from: My order #A-91 never arrived and support was rude. Refund me."}],
  "response_schema":{
    "type":"object",
    "properties":{
      "ticket":{
        "type":"object",
        "properties":{
          "order_id":{"type":"string"},
          "priority":{"type":"string","enum":["low","medium","high"]},
          "sentiment":{"type":"string","enum":["positive","neutral","negative"]}
        },
        "required":["order_id","priority","sentiment"],
        "additionalProperties":false
      },
      "issues":{
        "type":"array",
        "items":{
          "type":"object",
          "properties":{
            "category":{"type":"string","enum":["delivery","billing","support","product"]},
            "detail":{"type":"string"}
          },
          "required":["category","detail"],
          "additionalProperties":false
        }
      },
      "requested_action":{"type":"string"}
    },
    "required":["ticket","issues","requested_action"],
    "additionalProperties":false
  }
}' | jq .parsed
```

Returns a nested object like:

```json
{
  "ticket": {"order_id": "A-91", "priority": "high", "sentiment": "negative"},
  "issues": [
    {"category": "delivery", "detail": "order never arrived"},
    {"category": "support", "detail": "support was rude"}
  ],
  "requested_action": "refund"
}
```

Sample response:

```json
{
  "content": "{\"ticket\":{\"order_id\":\"A-91\",\"priority\":\"high\",\"sentiment\":\"negative\"},\"issues\":[{\"category\":\"delivery\",\"detail\":\"order never arrived\"},{\"category\":\"support\",\"detail\":\"support was rude\"}],\"requested_action\":\"refund\"}",
  "model": "gemini-3.6-flash",
  "provider": "gemini",
  "usage": {
    "input_tokens": 22,
    "output_tokens": 50
  },
  "latency_ms": 4722.132100003364,
  "finish_reason": "stop",
  "parsed": {
    "ticket": {
      "order_id": "A-91",
      "priority": "high",
      "sentiment": "negative"
    },
    "issues": [
      {
        "category": "delivery",
        "detail": "order never arrived"
      },
      {
        "category": "support",
        "detail": "support was rude"
      }
    ],
    "requested_action": "refund"
  }
}
```

How it works: the OpenAI family uses `response_format`, while Anthropic and
Bedrock (which have no such field) use *forced tool-use* - one tool whose input
schema is yours, forced via `tool_choice`, with the call's arguments as your
object. The adapters hide that difference.

**Validation is the guarantee.** Provider "strict" modes vary and some compat
layers ignore them, so the gateway validates the result against your schema
itself. A violation is treated like any other failed attempt - it **fails over
to the next provider**, because a different model may honor the contract. This is
a non-streaming feature (you want the whole validated object); streaming passes
the schema through where supported but isn't validated.

## Not built yet (next slices, in order)

1. Redis semantic cache + provider prompt caching.
2. Per-key token budgets + rate limiting.
3. Tracing (Langfuse / OpenTelemetry) + eval suite.