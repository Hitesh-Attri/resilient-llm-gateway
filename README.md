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
│   │   ├── types.py              # normalized request/response vocabulary
│   │   ├── provider.py           # Provider Protocol, ProviderError, classifiers
│   │   │                         #   (should_failover / is_transient)
│   │   ├── retry.py              # RetryPolicy + full-jitter backoff (pure)
│   │   ├── gateway.py            # the two loops: retry (inner) + fallback (outer)
│   │   ├── config.py             # config-driven chain + retry policy from env
│   │   ├── log_context.py        # getter and setter for request id context var
│   │   └── log.py                # get_logger(); logging setup lives here
│   ├── providers/                # one adapter per vendor (SDKs lazy-imported)
│   │   ├── openai_provider.py    # also serves Groq / Gemini / Ollama via base_url
│   │   ├── anthropic_provider.py
│   │   └── bedrock_provider.py
│   ├── api/
│   │   └── routes.py             # POST /v1/chat, GET /health
│   └── main.py                   # FastAPI app + `python src/main.py` entrypoint
├── tests/                        # no API keys needed (fake providers, injected clock)
│   ├── test_gateway_fallback.py  # outer loop: fail-over vs fail-fast
│   ├── test_error_classification.py  # should_failover / is_transient
│   └── test_retry.py             # inner loop: backoff schedule + retry-then-failover
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

## Not built yet (next slices, in order)

1. Streaming responses (SSE).
2. Structured outputs (Pydantic schema -> provider tool/JSON mode).
3. Redis semantic cache + provider prompt caching.
4. Per-key token budgets + rate limiting.
5. Tracing (Langfuse / OpenTelemetry) + eval suite.