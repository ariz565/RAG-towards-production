"""LLM Service — plug-and-play adapter for nine providers.

Bring your keys → set ACTIVE_MODEL in .env → everything works.

Providers : OpenAI · Azure OpenAI · OpenRouter · Anthropic · Groq
            NVIDIA NIM · Google Gemini · AWS Bedrock · Ollama

Sampling controls are centralised in config.py / .env.
Every provider has three layers of knobs:
  [Connection]   API keys, endpoints, timeout, retry
  [Sampling]     temperature, max_tokens, top_p, top_k, seed, stop, penalties
  [SDK-specific] reasoning_effort, thinking, extra_body, num_ctx, etc.

Reasoning-model awareness:
  OpenAI o-series     → temperature forced to 1, reasoning_effort injected
  Anthropic thinking  → opt-in via ANTHROPIC_THINKING=true
  NVIDIA DeepSeek     → opt-in via NVIDIA_THINKING=true (extra_body)

Output modes:
  chat()            plain text response
  chat_json()       provider-native JSON flag; raises ValueError on parse failure
  chat_messages()   pre-built message list (multi-turn, tool results)
  structured_chat() Pydantic-validated via .with_structured_output()
  stream_chat()     live token streaming via .astream()

Architecture notes:
  • @lru_cache singleton per (provider, model_name, temperature) — connection
    pool reused across all calls; no client created per request.
  • max_retries=0 on every model; our RetryPolicy is the sole retry authority.
  • ContextVar isolation — per-asyncio-task overrides, concurrency-safe.
"""

from __future__ import annotations

import contextvars
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import AsyncGenerator, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.config import ModelProvider, settings
from app.services.observability import (
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_TOTAL,
    gen_ai_attributes,
    span,
)
from app.services.retry import RetryPolicy, retry_async

logger = logging.getLogger(__name__)
_T = TypeVar("_T", bound=BaseModel)


# ═══════════════════════════════════════════════════════════════════════
# CONTEXT-VAR OVERRIDES  — per-asyncio-task, concurrency-safe
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
    """Reset overrides set by use_request_model()."""
    _provider_override.reset(tokens[0])
    _model_override_cv.reset(tokens[1])


def current_provider() -> ModelProvider:
    return _provider_override.get() or settings.active_model


def current_model_name() -> str:
    return _model_override_cv.get() or settings.model_name_for(current_provider())


# ═══════════════════════════════════════════════════════════════════════
# RESPONSE CONTAINER
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LLMResponse:
    """Uniform response from any provider."""
    content: str
    tokens_used: int
    model: str
    provider: str


# ═══════════════════════════════════════════════════════════════════════
# PROVIDER CAPABILITY SETS
# ═══════════════════════════════════════════════════════════════════════

# Providers accepting response_format={"type": "json_object"}
_OPENAI_COMPAT_PROVIDERS = frozenset({
    ModelProvider.OPENAI,
    ModelProvider.AZURE_OPENAI,
    ModelProvider.OPENROUTER,
    ModelProvider.GROQ,
    ModelProvider.NVIDIA,
})

# Providers that support the json_schema structured-output method
# (model enforces the schema at generation time — more reliable than json_object)
_JSON_SCHEMA_PROVIDERS = frozenset({
    ModelProvider.OPENAI,
    ModelProvider.AZURE_OPENAI,
    ModelProvider.GROQ,
})

# OpenAI o-series: require temperature=1, accept reasoning_effort
_OPENAI_REASONING_MODELS = frozenset({
    "o1", "o1-mini", "o1-preview", "o1-pro",
    "o3", "o3-mini", "o3-pro",
    "o4-mini",
})


def _is_openai_reasoning(model_name: str) -> bool:
    m = model_name.lower().split("/")[-1]
    base = m.rsplit("-20", 1)[0] if "-20" in m else m  # strip date suffix (o3-mini-2025-01-31 → o3-mini)
    return base in _OPENAI_REASONING_MODELS


# ═══════════════════════════════════════════════════════════════════════
# SAMPLING HELPERS
# ═══════════════════════════════════════════════════════════════════════


def _coalesce(value, fallback):
    """Return value if not None, else fallback (avoids falsy 0.0 pitfall)."""
    return value if value is not None else fallback


def _stop_sequences() -> list[str] | None:
    """Parse LLM_STOP_SEQUENCES from settings into a list, or None."""
    raw = settings.llm_stop_sequences.strip()
    if not raw:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


# ═══════════════════════════════════════════════════════════════════════
# SINGLETON MODEL FACTORY
#
# Cached per (provider, model_name, temperature).  Sampling overrides
# from settings are captured at first call and baked into the instance.
# Connection pools are reused across all requests with the same key.
# max_retries=0 on every model — RetryPolicy is the sole retry authority.
# ═══════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=64)
def _get_model(provider: ModelProvider, model_name: str, temperature: float):  # noqa: C901
    s = settings  # local alias for brevity
    timeout = s.llm_timeout_seconds
    stop = _stop_sequences()

    # ─────────────────────────────────────────────────────────────────
    # OpenAI  (langchain-openai ≥ 0.3 · openai SDK ≥ 1.10)
    # Models  : gpt-4o, gpt-4.1, o1, o3, o4-mini, …
    # Docs    : https://python.langchain.com/docs/integrations/chat/openai/
    # ─────────────────────────────────────────────────────────────────
    if provider == ModelProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        reasoning = _is_openai_reasoning(model_name)
        return ChatOpenAI(
            # — Connection —
            model=model_name,
            api_key=s.openai_api_key,
            organization=s.openai_organization or None,
            timeout=timeout,
            max_retries=0,
            # — Sampling —
            temperature=1.0 if reasoning else temperature,
            max_tokens=_coalesce(s.openai_max_tokens, s.llm_max_tokens),
            top_p=_coalesce(s.openai_top_p, s.llm_top_p),
            presence_penalty=s.openai_presence_penalty,
            frequency_penalty=s.openai_frequency_penalty,
            seed=s.llm_seed,
            stop=stop,
            # — Reasoning (o-series only; ignored for standard models) —
            model_kwargs={"reasoning_effort": s.openai_reasoning_effort} if reasoning else {},
        )

    # ─────────────────────────────────────────────────────────────────
    # Azure OpenAI  (langchain-openai · Azure SDK)
    # Models  : gpt-4o, gpt-4.1 deployed under your Azure resource
    # Docs    : https://python.langchain.com/docs/integrations/chat/azure_chat_openai/
    # ─────────────────────────────────────────────────────────────────
    if provider == ModelProvider.AZURE_OPENAI:
        from langchain_openai import AzureChatOpenAI
        reasoning = _is_openai_reasoning(model_name)
        return AzureChatOpenAI(
            # — Connection —
            azure_deployment=model_name,
            api_key=s.azure_openai_api_key,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            timeout=timeout,
            max_retries=0,
            # — Sampling (shares openai_* overrides) —
            temperature=1.0 if reasoning else temperature,
            max_tokens=_coalesce(s.openai_max_tokens, s.llm_max_tokens),
            top_p=_coalesce(s.openai_top_p, s.llm_top_p),
            presence_penalty=s.openai_presence_penalty,
            frequency_penalty=s.openai_frequency_penalty,
            seed=s.llm_seed,
            stop=stop,
            # — Reasoning —
            model_kwargs={"reasoning_effort": s.openai_reasoning_effort} if reasoning else {},
        )

    # ─────────────────────────────────────────────────────────────────
    # OpenRouter  (OpenAI-compatible gateway, 300+ models)
    # Models  : openai/gpt-4o, anthropic/claude-*, meta-llama/*, …
    # Docs    : https://openrouter.ai/docs
    # ─────────────────────────────────────────────────────────────────
    if provider == ModelProvider.OPENROUTER:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            # — Connection —
            model=model_name,
            api_key=s.openrouter_api_key,
            base_url=s.openrouter_base_url,
            timeout=timeout,
            max_retries=0,
            # — Sampling —
            temperature=temperature,
            max_tokens=_coalesce(s.openrouter_max_tokens, s.llm_max_tokens),
            top_p=_coalesce(s.openrouter_top_p, s.llm_top_p),
            seed=s.llm_seed,
            stop=stop,
        )

    # ─────────────────────────────────────────────────────────────────
    # Anthropic Claude  (langchain-anthropic ≥ 0.3 · anthropic SDK ≥ 0.96)
    # Models  : claude-opus-4-5, claude-3-7-sonnet, claude-haiku-*, …
    # Docs    : https://python.langchain.com/docs/integrations/chat/anthropic/
    # Extended thinking: set ANTHROPIC_THINKING=true in .env
    # ─────────────────────────────────────────────────────────────────
    if provider == ModelProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        thinking_cfg = (
            {"type": "enabled", "budget_tokens": s.anthropic_thinking_budget}
            if s.anthropic_thinking else None
        )
        kwargs: dict = dict(
            # — Connection —
            model=model_name,
            api_key=s.anthropic_api_key,
            timeout=timeout,
            max_retries=0,
            # — Sampling (extended thinking forces temperature=1) —
            temperature=1.0 if s.anthropic_thinking else temperature,
            max_tokens=_coalesce(s.anthropic_max_tokens, s.llm_max_tokens),
            stop_sequences=stop,
        )
        if s.anthropic_top_p is not None:
            kwargs["top_p"] = s.anthropic_top_p
        if s.anthropic_top_k is not None:
            kwargs["top_k"] = s.anthropic_top_k
        if thinking_cfg:
            kwargs["thinking"] = thinking_cfg
        return ChatAnthropic(**kwargs)

    # ─────────────────────────────────────────────────────────────────
    # Groq  (langchain-groq ≥ 0.2 · groq SDK)
    # Models  : llama-3.3-70b-versatile, mixtral-*, gemma-*, …
    # Docs    : https://python.langchain.com/docs/integrations/chat/groq/
    # ─────────────────────────────────────────────────────────────────
    if provider == ModelProvider.GROQ:
        from langchain_groq import ChatGroq
        return ChatGroq(
            # — Connection —
            model=model_name,
            api_key=s.groq_api_key,
            timeout=timeout,
            max_retries=0,
            # — Sampling —
            temperature=temperature,
            max_tokens=_coalesce(s.groq_max_tokens, s.llm_max_tokens),
            top_p=_coalesce(s.groq_top_p, s.llm_top_p),
            seed=s.llm_seed,
            stop_sequences=stop,
        )

    # ─────────────────────────────────────────────────────────────────
    # NVIDIA NIM  (langchain-nvidia-ai-endpoints ≥ 1.0)
    # Models  : deepseek-ai/deepseek-v4-pro, meta/llama-3.1-*, …
    # Docs    : https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/
    # Thinking: set NVIDIA_THINKING=true in .env (DeepSeek-R1 reasoning mode)
    # ─────────────────────────────────────────────────────────────────
    if provider == ModelProvider.NVIDIA:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        kwargs = dict(
            # — Connection —
            model=model_name,
            api_key=s.nvidia_api_key,
            # — Sampling —
            temperature=temperature,
            max_tokens=s.nvidia_max_tokens,
            top_p=s.nvidia_top_p,
        )
        if s.nvidia_thinking:
            # — SDK-specific: DeepSeek thinking toggle —
            kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": True}}
        if s.nvidia_top_k is not None:
            kwargs["top_k"] = s.nvidia_top_k
        return ChatNVIDIA(**kwargs)

    # ─────────────────────────────────────────────────────────────────
    # Google Gemini  (langchain-google-genai ≥ 2.0 · google-genai SDK)
    # Models  : gemini-2.0-flash, gemini-1.5-pro, gemini-2.5-flash, …
    # Docs    : https://python.langchain.com/docs/integrations/chat/google_generative_ai/
    # ─────────────────────────────────────────────────────────────────
    if provider == ModelProvider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = dict(
            # — Connection —
            model=model_name,
            google_api_key=s.gemini_api_key,
            timeout=timeout,
            max_retries=0,
            # — Sampling —
            temperature=temperature,
            max_output_tokens=_coalesce(s.gemini_max_tokens, s.llm_max_tokens),
            top_p=_coalesce(s.gemini_top_p, s.llm_top_p),
        )
        if s.gemini_top_k is not None:
            kwargs["top_k"] = s.gemini_top_k
        if stop:
            kwargs["stop_sequences"] = stop
        return ChatGoogleGenerativeAI(**kwargs)

    # ─────────────────────────────────────────────────────────────────
    # AWS Bedrock  (langchain-aws ≥ 0.2 · boto3)
    # Models  : us.anthropic.claude-*, amazon.titan-*, meta.llama-*, …
    # Auth    : AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or IAM role.
    # Docs    : https://python.langchain.com/docs/integrations/chat/bedrock/
    # Note    : ChatBedrockConverse is the recommended class for all modern models.
    # ─────────────────────────────────────────────────────────────────
    if provider == ModelProvider.BEDROCK:
        from langchain_aws import ChatBedrockConverse
        kwargs = dict(
            # — Connection —
            model=model_name,
            region_name=s.bedrock_region,
            # — Sampling —
            temperature=temperature,
            max_tokens=_coalesce(s.bedrock_max_tokens, s.llm_max_tokens),
        )
        if s.bedrock_top_p is not None:
            kwargs["top_p"] = s.bedrock_top_p
        if stop:
            kwargs["stop_sequences"] = stop
        return ChatBedrockConverse(**kwargs)

    # ─────────────────────────────────────────────────────────────────
    # Ollama  (langchain-ollama ≥ 0.3 · local server)
    # Models  : llama3.1, mistral, phi4, qwen2.5, deepseek-r1, …
    # Docs    : https://python.langchain.com/docs/integrations/chat/ollama/
    # ─────────────────────────────────────────────────────────────────
    if provider == ModelProvider.OLLAMA:
        from langchain_ollama import ChatOllama
        kwargs = dict(
            # — Connection —
            model=model_name,
            base_url=s.ollama_base_url,
            # — Sampling —
            temperature=temperature,
            num_predict=_coalesce(s.ollama_max_tokens, s.llm_max_tokens),
            top_p=_coalesce(s.ollama_top_p, s.llm_top_p),
            seed=s.llm_seed,
            stop=stop,
        )
        if s.ollama_top_k is not None:
            kwargs["top_k"] = s.ollama_top_k
        if s.ollama_repeat_penalty is not None:
            kwargs["repeat_penalty"] = s.ollama_repeat_penalty
        if s.ollama_num_ctx is not None:
            kwargs["num_ctx"] = s.ollama_num_ctx
        return ChatOllama(**kwargs)

    raise ValueError(f"Unknown provider: {provider}")  # pragma: no cover


# ═══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════


def _resolve(
    provider: ModelProvider | None,
    model_override: str | None,
    temperature: float | None,
) -> tuple[ModelProvider, str, float]:
    prov = provider or _provider_override.get() or settings.active_model
    name = model_override or _model_override_cv.get() or settings.model_name_for(prov)
    temp = temperature if temperature is not None else settings.temperature
    return prov, name, temp


def _build_messages(prompt: str, system: str | None) -> list:
    msgs: list = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=prompt))
    return msgs


def _to_lc_messages(messages: list[dict]) -> list:
    """Convert OpenAI-style message dicts to LangChain message objects."""
    result: list = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role in ("assistant", "ai"):
            result.append(AIMessage(content=content))
        else:
            result.append(HumanMessage(content=content))
    return result


def _bind_json_mode(model, provider: ModelProvider):
    """Bind provider-native JSON output flag.

    Anthropic and Bedrock have no JSON flag; those callers rely on
    prompt instructions plus the parse-failure raise in chat_json().
    """
    if provider in _OPENAI_COMPAT_PROVIDERS:
        return model.bind(response_format={"type": "json_object"})
    if provider == ModelProvider.GEMINI:
        return model.bind(response_mime_type="application/json")
    if provider == ModelProvider.OLLAMA:
        return model.bind(format="json")
    return model  # Anthropic, Bedrock: prompt-guided


def _extract_response(ai_msg: AIMessage, model_name: str, provider_name: str) -> LLMResponse:
    """Normalise a LangChain AIMessage into LLMResponse.

    langchain-core ≥ 0.2 exposes ai_msg.usage_metadata with
    input_tokens / output_tokens / total_tokens across all providers.
    """
    content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
    usage: dict = getattr(ai_msg, "usage_metadata", None) or {}
    tokens = usage.get("total_tokens") or (
        usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    )
    return LLMResponse(content=content, tokens_used=tokens or 0, model=model_name, provider=provider_name)


def _retry_policy() -> RetryPolicy:
    s = settings
    return RetryPolicy(
        max_retries=s.llm_max_retries,
        base_delay=s.llm_retry_base_delay,
        max_delay=s.llm_retry_max_delay,
        jitter=s.llm_retry_jitter,
        respect_retry_after=s.llm_retry_respect_retry_after,
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
    """Single-turn text chat.

    Returns LLMResponse(content, tokens_used, model, provider).
    All sampling controls are resolved from config/env; override temperature
    here for one-off call-site needs.
    """
    prov, model_name, temp = _resolve(provider, model_override, temperature)
    model = _get_model(prov, model_name, temp)
    messages = _build_messages(prompt, system)

    with span("gen_ai.chat", gen_ai_attributes(prov.value, model_name)) as sp:
        ai_msg = await retry_async(
            lambda: model.ainvoke(messages),
            policy=_retry_policy(),
            label=f"LLM[{prov.value}/{model_name}]",
        )
        response = _extract_response(ai_msg, model_name, prov.value)
        sp.set_attribute(GEN_AI_RESPONSE_MODEL, response.model)
        sp.set_attribute(GEN_AI_USAGE_TOTAL, response.tokens_used)

    logger.debug("LLM [%s/%s] %d tokens", response.provider, response.model, response.tokens_used)
    return response


async def chat_json(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    provider: ModelProvider | None = None,
    model_override: str | None = None,
) -> tuple[dict, int]:
    """Chat with provider-native JSON mode. Raises ValueError on parse failure.

    JSON-mode routing:
      OpenAI / Azure / Groq / NVIDIA / OpenRouter → response_format json_object
      Gemini                                       → response_mime_type application/json
      Ollama                                       → format json
      Anthropic / Bedrock                          → prompt-guided (no native flag)

    Returns (parsed_dict, tokens_used).
    """
    prov, model_name, temp = _resolve(provider, model_override, temperature)
    json_model = _bind_json_mode(_get_model(prov, model_name, temp), prov)
    messages = _build_messages(prompt, system)
    label = f"LLM JSON [{prov.value}/{model_name}]"

    with span("gen_ai.chat_json", gen_ai_attributes(prov.value, model_name)) as sp:
        ai_msg = await retry_async(
            lambda: json_model.ainvoke(messages),
            policy=_retry_policy(),
            label=label,
        )
        response = _extract_response(ai_msg, model_name, prov.value)
        sp.set_attribute(GEN_AI_RESPONSE_MODEL, response.model)
        sp.set_attribute(GEN_AI_USAGE_TOTAL, response.tokens_used)

    logger.debug("LLM JSON [%s/%s] %d tokens", response.provider, response.model, response.tokens_used)

    content = response.content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(content), response.tokens_used
    except json.JSONDecodeError as exc:
        logger.error("LLM JSON parse failure [%s]: %.300s", label, content)
        raise ValueError(f"LLM did not return valid JSON: {exc}") from exc


async def chat_messages(
    messages: list[dict],
    *,
    json_mode: bool = False,
    temperature: float | None = None,
    provider: ModelProvider | None = None,
    model_override: str | None = None,
) -> LLMResponse:
    """Send a pre-built OpenAI-style message list.

    Use for multi-turn conversations, tool results, or any call site that
    needs full control over the message history.
    """
    prov, model_name, temp = _resolve(provider, model_override, temperature)
    base_model = _get_model(prov, model_name, temp)
    model = _bind_json_mode(base_model, prov) if json_mode else base_model
    lc_messages = _to_lc_messages(messages)

    with span("gen_ai.chat", gen_ai_attributes(prov.value, model_name)) as sp:
        ai_msg = await retry_async(
            lambda: model.ainvoke(lc_messages),
            policy=_retry_policy(),
            label=f"LLM messages [{prov.value}/{model_name}]",
        )
        response = _extract_response(ai_msg, model_name, prov.value)
        sp.set_attribute(GEN_AI_RESPONSE_MODEL, response.model)
        sp.set_attribute(GEN_AI_USAGE_TOTAL, response.tokens_used)

    return response


async def structured_chat(
    prompt: str,
    *,
    schema: type[_T],
    system: str | None = None,
    temperature: float | None = None,
    provider: ModelProvider | None = None,
    model_override: str | None = None,
) -> _T:
    """Return a Pydantic-validated instance via .with_structured_output().

    Structured-output method selection:
      OpenAI / Azure / Groq  → json_schema  (model enforces schema at generation time)
      All others             → LangChain default (function_calling or json_mode)

    include_raw=True is set internally so token counts are preserved.

    Raises ValueError if the model response cannot be parsed into schema.

    Example::

        class Guardrail(BaseModel):
            score: int
            reasoning: str

        result = await structured_chat("Score this", schema=Guardrail, temperature=0)
        print(result.score)
    """
    prov, model_name, temp = _resolve(provider, model_override, temperature)
    base_model = _get_model(prov, model_name, temp)

    use_json_schema = prov in _JSON_SCHEMA_PROVIDERS and not _is_openai_reasoning(model_name)
    structured = (
        base_model.with_structured_output(schema, method="json_schema", include_raw=True)
        if use_json_schema
        else base_model.with_structured_output(schema, include_raw=True)
    )
    messages = _build_messages(prompt, system)
    label = f"LLM structured [{prov.value}/{model_name}]"

    with span("gen_ai.structured_chat", gen_ai_attributes(prov.value, model_name)) as sp:
        raw: dict = await retry_async(
            lambda: structured.ainvoke(messages),
            policy=_retry_policy(),
            label=label,
        )
        parsed = raw.get("parsed")
        if parsed is None:
            raise ValueError(f"Structured output failed [{label}]: {raw.get('parsing_error')}")

        usage: dict = getattr(raw.get("raw"), "usage_metadata", None) or {}
        tokens = usage.get("total_tokens") or (
            usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        )
        sp.set_attribute(GEN_AI_RESPONSE_MODEL, model_name)
        sp.set_attribute(GEN_AI_USAGE_TOTAL, tokens or 0)

    logger.debug("LLM structured [%s/%s] %d tokens", prov.value, model_name, tokens or 0)
    return parsed


async def stream_chat(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    provider: ModelProvider | None = None,
    model_override: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield content chunks via model.astream() for real-time rendering.

    Retry is not applied — a stream cannot be restarted mid-flight.
    """
    prov, model_name, temp = _resolve(provider, model_override, temperature)
    model = _get_model(prov, model_name, temp)
    messages = _build_messages(prompt, system)

    logger.debug("LLM stream [%s/%s] starting", prov.value, model_name)
    async for chunk in model.astream(messages):
        if chunk.content:
            yield str(chunk.content)


# ═══════════════════════════════════════════════════════════════════════
# PROVIDER STATUS
# ═══════════════════════════════════════════════════════════════════════


def get_available_providers() -> list[dict]:
    """Return all providers with their active model and credential status."""
    s = settings
    return [
        {
            "id": ModelProvider.OPENAI.value,
            "name": "OpenAI",
            "model": s.openai_model,
            "configured": bool(s.openai_api_key),
            "local": False,
        },
        {
            "id": ModelProvider.AZURE_OPENAI.value,
            "name": "Azure OpenAI",
            "model": s.azure_openai_deployment,
            "configured": bool(s.azure_openai_api_key and s.azure_openai_endpoint),
            "local": False,
        },
        {
            "id": ModelProvider.OPENROUTER.value,
            "name": "OpenRouter",
            "model": s.openrouter_model,
            "configured": bool(s.openrouter_api_key),
            "local": False,
        },
        {
            "id": ModelProvider.ANTHROPIC.value,
            "name": "Anthropic Claude",
            "model": s.anthropic_model,
            "configured": bool(s.anthropic_api_key),
            "local": False,
        },
        {
            "id": ModelProvider.GROQ.value,
            "name": "Groq",
            "model": s.groq_model,
            "configured": bool(s.groq_api_key),
            "local": False,
        },
        {
            "id": ModelProvider.NVIDIA.value,
            "name": "NVIDIA NIM",
            "model": s.nvidia_model,
            "configured": bool(s.nvidia_api_key),
            "local": False,
        },
        {
            "id": ModelProvider.GEMINI.value,
            "name": "Google Gemini",
            "model": s.gemini_model,
            "configured": bool(s.gemini_api_key),
            "local": False,
        },
        {
            "id": ModelProvider.BEDROCK.value,
            "name": "AWS Bedrock",
            "model": s.bedrock_model,
            "configured": True,  # uses boto3 credential chain; no static key to validate
            "local": False,
        },
        {
            "id": ModelProvider.OLLAMA.value,
            "name": "Ollama (Local)",
            "model": s.ollama_model,
            "configured": True,
            "local": True,
        },
    ]
