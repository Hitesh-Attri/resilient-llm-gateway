"""Config-driven wiring. The fallback chain is defined by a single env string:

    LLM_CHAIN="openai:gpt-5-mini,anthropic:claude-sonnet-4-5,bedrock:us.anthropic.claude-sonnet-4-5-v1:0"

The factory parses that into Target objects, constructing each provider exactly
once (a provider client is reused even if it appears at multiple rungs). This is
the same config-over-code instinct behind your Integration Framework: changing
the fallback order is a config change, not a code change.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from core.gateway import LLMGateway, Target
from core.provider import Provider
from core.retry import RetryPolicy


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ordered, comma-separated "provider:model" rungs. First is primary.
    llm_chain: str = "anthropic:claude-sonnet-4-5"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    aws_region: str = "ap-south-1"

    # OpenAI-compatible providers (free tiers - useful for learning and as
    # genuine fallback lanes).
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434/v1"

    # Retry policy for the inner loop (per-target, on transient failures).
    retry_max_attempts: int = 3     # 1 initial try + 2 retries
    retry_base_delay: float = 0.5   # seconds
    retry_max_delay: float = 8.0    # cap per backoff sleep


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _require(value: str | None, name: str, key_env: str) -> str:
    if not value:
        raise ValueError(f"{name} in chain but {key_env} is not set")
    return value


def _construct_provider(name: str, settings: Settings) -> Provider:
    """Build a provider by name. SDK imports live inside the adapter modules and
    are only triggered here, so the core package stays importable without any
    vendor SDK installed. Each provider is its own class, configured individually."""
    if name == "openai":
        from providers.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=_require(settings.openai_api_key, name, "OPENAI_API_KEY"))

    if name == "groq":
        from providers.groq_provider import GroqProvider

        return GroqProvider(api_key=_require(settings.groq_api_key, name, "GROQ_API_KEY"))

    if name == "gemini":
        from providers.gemini_provider import GeminiProvider

        return GeminiProvider(api_key=_require(settings.gemini_api_key, name, "GEMINI_API_KEY"))

    if name == "ollama":
        from providers.ollama_provider import OllamaProvider

        # Local models: no key needed, but the SDK requires a non-empty string.
        return OllamaProvider(api_key="ollama", base_url=settings.ollama_base_url)

    if name == "anthropic":
        from providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=_require(settings.anthropic_api_key, name, "ANTHROPIC_API_KEY")
        )

    if name == "bedrock":
        from providers.bedrock_provider import BedrockProvider

        return BedrockProvider(region=settings.aws_region)

    raise ValueError(f"unknown provider '{name}' in LLM_CHAIN")


def build_gateway(settings: Settings | None = None) -> LLMGateway:
    settings = settings or get_settings()
    cache: dict[str, Provider] = {}
    targets: list[Target] = []

    for rung in settings.llm_chain.split(","):
        name, _, model = rung.strip().partition(":")
        name, model = name.strip(), model.strip()
        if not name or not model:
            raise ValueError(f"malformed chain rung '{rung}', expected 'provider:model'")
        if name not in cache:
            cache[name] = _construct_provider(name, settings)
        targets.append(Target(provider=cache[name], model=model))

    policy = RetryPolicy(
        max_attempts=settings.retry_max_attempts,
        base_delay=settings.retry_base_delay,
        max_delay=settings.retry_max_delay,
    )
    return LLMGateway(targets, retry=policy)