# LLM Gateway (Module 1)

A from-scratch, provider-agnostic gateway that fronts OpenAI, Anthropic, and AWS
Bedrock behind one interface, with an ordered fallback chain. This is the
foundation every later module (RAG, agent, MCP server) makes its model calls
through.

## The idea in one line

Callers speak one normalized vocabulary (`ChatRequest` / `ChatResponse`).
Adapters translate to each vendor. The gateway tries targets in order and fails
over on transient errors, fails fast on bad requests.

## Layout

```
llm-gateway/
├── src/                     # import root (see "Imports" below)
│   ├── core/
│   │   ├── types.py         # normalized request/response vocabulary
│   │   ├── provider.py      # Provider Protocol + ProviderError(retryable=...)
│   │   ├── gateway.py       # the fallback chain (the heart)
│   │   └── config.py        # config-driven chain from LLM_CHAIN env
│   ├── providers/           # one adapter per vendor (SDKs lazy-imported)
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   └── bedrock_provider.py
│   ├── api/
│   │   └── routes.py        # POST /v1/chat
│   └── main.py              # FastAPI app + `python src/main.py` entrypoint
├── tests/                   # fallback tests (no API keys needed)
├── infra/                   # OpenTofu + Terragrunt (deployment slice, later)
├── .github/workflows/ci.yml # lint + test on push/PR
├── Dockerfile
├── docker-compose.yml       # gateway + redis (redis staged for cache slice)
├── pyproject.toml           # deps + makes src/ the import root
├── requirements.txt         # runtime deps (for Docker layer caching)
└── requirements-dev.txt
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

## Test

```bash
pytest                           # 5 fallback tests, no API keys needed
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

## Not built yet (next slices, in order)

1. Per-target retries with exponential backoff + jitter (before failing over).
2. Streaming responses (SSE).
3. Structured outputs (Pydantic schema -> provider tool/JSON mode).
4. Redis semantic cache + provider prompt caching.
5. Per-key token budgets + rate limiting.
6. Tracing (Langfuse / OpenTelemetry) + eval suite.