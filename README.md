# LLM Gateway (Module 1)

A from-scratch, provider-agnostic gateway that fronts OpenAI, Anthropic, and AWS
Bedrock behind one interface, with an ordered fallback chain. This is the
foundation every later module (RAG, agent, MCP server) makes its model calls
through.

## The idea in one line

Callers speak one normalized vocabulary (`ChatRequest` / `ChatResponse`).
Adapters translate to each vendor. The gateway tries targets in order and fails
over on transient errors, fails fast on bad requests.

## Architecture

```
        POST /v1/chat
             |
        api/routes.py        (thin HTTP layer: validate -> delegate -> map errors)
             |
        core/gateway.py      (fallback chain: try, classify failure, fail-over vs fail-fast)
             |
   +---------+-----------------------+
   |         |                       |
providers/ openai  anthropic  bedrock  (adapters: normalize each vendor's API)
```

- `core/types.py` - the normalized request/response vocabulary.
- `core/provider.py` - the `Provider` Protocol + the `ProviderError(retryable=...)` that drives fallback.
- `core/gateway.py` - the fallback orchestrator (the heart).
- `core/config.py` - config-driven chain: `LLM_CHAIN="openai:...,anthropic:...,bedrock:..."`.
- `providers/*` - one adapter per vendor; SDKs are lazy-imported.
- `api/routes.py`, `main.py` - FastAPI surface.

## Run it

```bash
cd llm-gateway
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # set keys + confirm current model IDs
uvicorn app.main:app --reload
```

Then:

```bash
curl -s localhost:8000/v1/chat -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"Explain a token bucket in one sentence."}],
  "system":"You are terse."
}' | jq
```

## Test

```bash
python -m pytest      # 5 fallback tests, no API keys needed
```

The tests use fake providers because the gateway depends only on the `Provider`
protocol - a decoupling that also means you can add a new vendor without touching
the gateway or the tests.

## Design decisions worth knowing

- **System prompt is a separate field**, not a message. Anthropic/Bedrock want it
  top-level; OpenAI wants it as a `role: system` message. Keeping it separate lets
  each adapter place it correctly.
- **Fail-over vs fail-fast** is decided by `ProviderError.retryable`. Transient
  (429 / 5xx / timeout / connection) fails over to the next target. A malformed or
  unauthorized request (4xx) fails fast - another provider would reject it too.
- **Lazy SDK imports.** The core imports with zero vendor SDKs installed; you only
  install the SDKs for providers actually in your chain.

## Not built yet (next slices)

Intentionally scoped out of this first slice, in build order:

1. Per-target retries with exponential backoff + jitter (before failing over).
2. Streaming responses (SSE) - `ChatResponse` becomes an async token generator.
3. Structured outputs (Pydantic schema -> provider tool/JSON mode).
4. Redis semantic cache + provider prompt caching.
5. Per-key token budgets + rate limiting.
6. Tracing (Langfuse / OpenTelemetry) with cost-per-request, and an eval suite.