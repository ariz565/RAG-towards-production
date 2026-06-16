"""Observability (Phase D) — OpenTelemetry GenAI semantic conventions.

A thin tracing facade that emits ``gen_ai.*`` spans for LLM calls plus
request/node spans for the pipeline. It is **optional and safe**:

- If ``OTEL_ENABLED`` is false or OpenTelemetry isn't installed, every span is a
  no-op (zero overhead, no dependency required).
- When enabled, spans export via OTLP (Langfuse, Datadog, Honeycomb, …) and/or
  the console — using the 2026 OTel GenAI semantic-convention attribute names.

This keeps the project standards-based and vendor-neutral.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# GenAI semantic-convention attribute keys (OTel gen_ai.*)
# ═══════════════════════════════════════════════════════════════════════

GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_TOTAL = "gen_ai.usage.total_tokens"


# ═══════════════════════════════════════════════════════════════════════
# NO-OP SPAN — lets callers always use the same API
# ═══════════════════════════════════════════════════════════════════════


class _NoopSpan:
    def set_attribute(self, *_args, **_kwargs) -> None:
        pass

    def set_attributes(self, *_args, **_kwargs) -> None:
        pass

    def record_exception(self, *_args, **_kwargs) -> None:
        pass

    def set_status(self, *_args, **_kwargs) -> None:
        pass

    def add_event(self, *_args, **_kwargs) -> None:
        pass


_NOOP = _NoopSpan()


# ═══════════════════════════════════════════════════════════════════════
# TRACER INITIALIZATION (lazy, optional)
# ═══════════════════════════════════════════════════════════════════════

_tracer = None
_initialized = False


def _init_tracer():
    """Initialize the OTel tracer once. Returns a tracer or None."""
    global _tracer, _initialized
    if _initialized:
        return _tracer
    _initialized = True

    if not settings.otel_enabled:
        logger.info("Observability: disabled (spans are no-ops).")
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(resource=resource)

        if settings.otel_exporter_otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
                )
            )
            logger.info(f"Observability: OTLP → {settings.otel_exporter_otlp_endpoint}")

        if settings.otel_console_export:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logger.info("Observability: console span export enabled")

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("vision")
        return _tracer
    except Exception as e:  # opentelemetry not installed / misconfigured
        logger.warning(f"Observability: OTel unavailable ({e}); spans are no-ops.")
        _tracer = None
        return None


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════


@contextmanager
def span(name: str, attributes: dict | None = None):
    """Start a span (no-op if observability is disabled/unavailable)."""
    tracer = _init_tracer()
    if tracer is None:
        if attributes:
            pass  # nothing to record
        yield _NOOP
        return
    with tracer.start_as_current_span(name) as sp:
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    sp.set_attribute(k, v)
        try:
            yield sp
        except Exception as e:  # record and re-raise
            sp.record_exception(e)
            raise


def gen_ai_attributes(provider: str, model: str, operation: str = "chat") -> dict:
    """Build the standard gen_ai.* request attributes for an LLM call."""
    return {
        GEN_AI_OPERATION: operation,
        GEN_AI_SYSTEM: provider,
        GEN_AI_REQUEST_MODEL: model,
    }
