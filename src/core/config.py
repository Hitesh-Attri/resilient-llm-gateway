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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ordered, comma-separated "provider:model" rungs. First is primary.
    llm_chain: str = "anthropic:claude-sonnet-4-5"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    aws_region: str = "ap-south-1"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _construct_provider(name: str, settings: Settings) -> Provider:
    """Build a provider by name. SDK imports live inside the adapter modules and
    are only triggered here, so the core package stays importable without any
    vendor SDK installed."""
    if name == "openai":
        from providers.openai_provider import OpenAIProvider

        if not settings.openai_api_key:
            raise ValueError("openai in chain but OPENAI_API_KEY is not set")
        return OpenAIProvider(api_key=settings.openai_api_key)

    if name == "anthropic":
        from providers.anthropic_provider import AnthropicProvider

        if not settings.anthropic_api_key:
            raise ValueError("anthropic in chain but ANTHROPIC_API_KEY is not set")
        return AnthropicProvider(api_key=settings.anthropic_api_key)

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

    return LLMGateway(targets)