"""Centralized LLM Service — one interface, many providers.

Switch between OpenAI, Azure OpenAI, OpenRouter, Groq, Google Gemini,
and Ollama (local)
with a single config flag. Every pipeline node calls this instead of a
provider-specific SDK.

Usage:
    from app.services.llm import chat, chat_json

    result = await chat("What is the GPA requirement?")
    data = await chat_json("Score this query", json_schema_hint="...")
"""

from __future__ import annotations

import contextvars
import json
import logging
from dataclasses import dataclass

from app.config import ModelProvider, settings
from app.services.observability import (
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_TOTAL,
    gen_ai_attributes,
    span,
)
from app.services.retry import RetryPolicy, retry_async

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# PER-REQUEST OVERRIDES (concurrency-safe)
# A request may select a provider/model without mutating global settings.
# ContextVars are isolated per asyncio task, so concurrent requests with
# different providers never stomp each other.
# ═══════════════════════════════════════════════════════════════════════

_provider_override: contextvars.ContextVar[ModelProvider | None] = contextvars.ContextVar(
    "vision_provider_override", default=None
)
_model_override_cv: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "vision_model_override", default=None
)


def use_request_model(provider: ModelProvider | None, model: str | None):
    """Set per-request provider/model overrides. Returns reset tokens."""
    return (_provider_override.set(provider), _model_override_cv.set(model))


def clear_request_model(tokens) -> None:
    """Reset the overrides set by use_request_model()."""
    provider_token, model_token = tokens
    _provider_override.reset(provider_token)
    _model_override_cv.reset(model_token)


def current_provider() -> ModelProvider:
    return _provider_override.get() or settings.active_model


def current_model_name() -> str:
    return _model_override_cv.get() or settings.model_name_for(current_provider())


# ═══════════════════════════════════════════════════════════════════════
# RESPONSE CONTAINER
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class LLMResponse:
    """Uniform response from any LLM provider."""
    content: str
    tokens_used: int
    model: str
    provider: str


# ═══════════════════════════════════════════════════════════════════════
# PROVIDER IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════


async def _openai_compatible_chat(
    messages: list[dict],
    temperature: float,
    json_mode: bool,
    *,
    provider_name: str,
    api_key: str,
    base_url: str | None,
    default_model: str,
    model_override: str | None = None,
) -> LLMResponse:
    """Call an OpenAI-compatible chat completion API."""
    from openai import AsyncOpenAI

    client_kwargs = {"api_key": api_key, "timeout": settings.llm_timeout_seconds}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)
    model = model_override or default_model

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)

    return LLMResponse(
        content=response.choices[0].message.content,
        tokens_used=response.usage.total_tokens if response.usage else 0,
        model=model,
        provider=provider_name,
    )


async def _openai_chat(
    messages: list[dict],
    temperature: float,
    json_mode: bool,
    model_override: str | None = None,
) -> LLMResponse:
    """Call OpenAI API."""
    return await _openai_compatible_chat(
        messages,
        temperature,
        json_mode,
        provider_name="openai",
        api_key=settings.openai_api_key,
        base_url=None,
        default_model=settings.openai_model,
        model_override=model_override,
    )


async def _azure_openai_chat(
    messages: list[dict],
    temperature: float,
    json_mode: bool,
    model_override: str | None = None,
) -> LLMResponse:
    """Call Azure OpenAI API."""
    from openai import AsyncAzureOpenAI

    deployment = model_override or settings.azure_openai_deployment
    client = AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        timeout=settings.llm_timeout_seconds,
    )

    kwargs = {
        "model": deployment,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)

    return LLMResponse(
        content=response.choices[0].message.content,
        tokens_used=response.usage.total_tokens if response.usage else 0,
        model=deployment,
        provider="azure_openai",
    )


async def _openrouter_chat(
    messages: list[dict],
    temperature: float,
    json_mode: bool,
    model_override: str | None = None,
) -> LLMResponse:
    """Call OpenRouter API (OpenAI-compatible)."""
    return await _openai_compatible_chat(
        messages,
        temperature,
        json_mode,
        provider_name="openrouter",
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_model=settings.openrouter_model,
        model_override=model_override,
    )


async def _groq_chat(
    messages: list[dict],
    temperature: float,
    json_mode: bool,
    model_override: str | None = None,
) -> LLMResponse:
    """Call Groq API (OpenAI-compatible)."""
    return await _openai_compatible_chat(
        messages,
        temperature,
        json_mode,
        provider_name="groq",
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
        default_model=settings.groq_model,
        model_override=model_override,
    )


async def _gemini_chat(
    messages: list[dict],
    temperature: float,
    json_mode: bool,
    model_override: str | None = None,
) -> LLMResponse:
    """Call Google Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    model_name = model_override or settings.gemini_model
    model = genai.GenerativeModel(model_name)

    # Convert OpenAI-style messages to Gemini format
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append({"role": "user", "parts": [content]})
            parts.append({"role": "model", "parts": ["Understood. I'll follow those instructions."]})
        else:
            gemini_role = "model" if role == "assistant" else "user"
            parts.append({"role": gemini_role, "parts": [content]})

    generation_config = {"temperature": temperature}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"

    response = await model.generate_content_async(
        parts,
        generation_config=generation_config,
    )

    # Estimate tokens (Gemini doesn't always return exact counts)
    content = response.text
    tokens = getattr(response, "usage_metadata", None)
    token_count = 0
    if tokens:
        token_count = getattr(tokens, "total_token_count", 0)

    return LLMResponse(
        content=content,
        tokens_used=token_count,
        model=model_name,
        provider="gemini",
    )


async def _ollama_chat(
    messages: list[dict],
    temperature: float,
    json_mode: bool,
    model_override: str | None = None,
) -> LLMResponse:
    """Call Ollama local API."""
    import ollama

    model = model_override or settings.ollama_model
    client = ollama.AsyncClient(host=settings.ollama_base_url, timeout=settings.llm_timeout_seconds)

    kwargs = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
    }
    if json_mode:
        kwargs["format"] = "json"

    response = await client.chat(**kwargs)

    content = response.get("message", {}).get("content", "")
    tokens = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)

    return LLMResponse(
        content=content,
        tokens_used=tokens,
        model=model,
        provider="ollama",
    )


# ═══════════════════════════════════════════════════════════════════════
# DISPATCH TABLE
# ═══════════════════════════════════════════════════════════════════════

_PROVIDERS = {
    ModelProvider.OPENAI: _openai_chat,
    ModelProvider.AZURE_OPENAI: _azure_openai_chat,
    ModelProvider.OPENROUTER: _openrouter_chat,
    ModelProvider.GROQ: _groq_chat,
    ModelProvider.GEMINI: _gemini_chat,
    ModelProvider.OLLAMA: _ollama_chat,
}


def _llm_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_retries=settings.llm_max_retries,
        base_delay=settings.llm_retry_base_delay,
        max_delay=settings.llm_retry_max_delay,
        jitter=settings.llm_retry_jitter,
        respect_retry_after=settings.llm_retry_respect_retry_after,
    )


async def _call_with_retry(fn, messages, temperature, json_mode, model_override):
    """Call a provider under the professional retry policy (transient-only)."""
    return await retry_async(
        lambda: fn(messages, temperature, json_mode=json_mode, model_override=model_override),
        policy=_llm_retry_policy(),
        label=f"LLM[{fn.__name__}]",
    )


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════


async def chat(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    provider: ModelProvider | None = None,
    model_override: str | None = None,
) -> LLMResponse:
    """Send a chat message to the active LLM provider.

    Args:
        prompt: The user message.
        system: Optional system message.
        temperature: Override default temperature.
        provider: Override active_model from settings.
        model_override: Override the model name.

    Returns:
        LLMResponse with content, tokens, model, provider.
    """
    provider = provider or _provider_override.get() or settings.active_model
    model_override = model_override or _model_override_cv.get()
    temp = temperature if temperature is not None else settings.temperature

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    fn = _PROVIDERS[provider]
    req_model = model_override or settings.model_name_for(provider)
    with span("gen_ai.chat", gen_ai_attributes(provider.value, req_model)) as sp:
        response = await _call_with_retry(fn, messages, temp, False, model_override)
        sp.set_attribute(GEN_AI_RESPONSE_MODEL, response.model)
        sp.set_attribute(GEN_AI_USAGE_TOTAL, response.tokens_used)

    logger.debug(
        f"LLM [{response.provider}/{response.model}] "
        f"{response.tokens_used} tokens"
    )
    return response


async def chat_json(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    provider: ModelProvider | None = None,
    model_override: str | None = None,
) -> tuple[dict, int]:
    """Send a chat message and parse JSON response.

    Returns:
        (parsed_dict, tokens_used)
    """
    provider = provider or _provider_override.get() or settings.active_model
    model_override = model_override or _model_override_cv.get()
    temp = temperature if temperature is not None else settings.temperature

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    fn = _PROVIDERS[provider]
    req_model = model_override or settings.model_name_for(provider)
    with span("gen_ai.chat", gen_ai_attributes(provider.value, req_model)) as sp:
        response = await _call_with_retry(fn, messages, temp, True, model_override)
        sp.set_attribute(GEN_AI_RESPONSE_MODEL, response.model)
        sp.set_attribute(GEN_AI_USAGE_TOTAL, response.tokens_used)

    logger.debug(
        f"LLM JSON [{response.provider}/{response.model}] "
        f"{response.tokens_used} tokens"
    )

    # Parse JSON — handle potential markdown code blocks
    content = response.content.strip()
    if content.startswith("```"):
        # Strip markdown code fences
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from LLM: {content[:200]}")
        data = {"error": "Failed to parse JSON", "raw": content}

    return data, response.tokens_used


async def chat_messages(
    messages: list[dict],
    *,
    json_mode: bool = False,
    temperature: float | None = None,
    provider: ModelProvider | None = None,
    model_override: str | None = None,
) -> LLMResponse:
    """Send pre-built messages to the active LLM.

    For advanced use cases where you need full control over the message list.
    """
    provider = provider or _provider_override.get() or settings.active_model
    model_override = model_override or _model_override_cv.get()
    temp = temperature if temperature is not None else settings.temperature

    fn = _PROVIDERS[provider]
    req_model = model_override or settings.model_name_for(provider)
    with span("gen_ai.chat", gen_ai_attributes(provider.value, req_model)) as sp:
        response = await _call_with_retry(fn, messages, temp, json_mode, model_override)
        sp.set_attribute(GEN_AI_RESPONSE_MODEL, response.model)
        sp.set_attribute(GEN_AI_USAGE_TOTAL, response.tokens_used)
    return response


def get_available_providers() -> list[dict]:
    """Return list of configured providers with status."""
    providers = []

    # OpenAI
    providers.append({
        "id": ModelProvider.OPENAI.value,
        "name": "OpenAI",
        "model": settings.openai_model,
        "configured": bool(settings.openai_api_key),
        "local": False,
    })

    # Azure OpenAI
    providers.append({
        "id": ModelProvider.AZURE_OPENAI.value,
        "name": "Azure OpenAI",
        "model": settings.azure_openai_deployment,
        "configured": bool(settings.azure_openai_api_key and settings.azure_openai_endpoint),
        "local": False,
    })

    # OpenRouter
    providers.append({
        "id": ModelProvider.OPENROUTER.value,
        "name": "OpenRouter",
        "model": settings.openrouter_model,
        "configured": bool(settings.openrouter_api_key),
        "local": False,
    })

    # Groq
    providers.append({
        "id": ModelProvider.GROQ.value,
        "name": "Groq",
        "model": settings.groq_model,
        "configured": bool(settings.groq_api_key),
        "local": False,
    })

    # Gemini
    providers.append({
        "id": ModelProvider.GEMINI.value,
        "name": "Google Gemini",
        "model": settings.gemini_model,
        "configured": bool(settings.gemini_api_key),
        "local": False,
    })

    # Ollama
    providers.append({
        "id": ModelProvider.OLLAMA.value,
        "name": "Ollama (Local)",
        "model": settings.ollama_model,
        "configured": True,  # Always available if Ollama is running
        "local": True,
    })

    return providers
