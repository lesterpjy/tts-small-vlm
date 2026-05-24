"""Phoenix / OpenTelemetry tracing setup.

Vendor-specific Phoenix / OTel code lives in this module. The rest of the
codebase consumes a plain `opentelemetry.trace.Tracer` via
`RunContext.stage_span` / `RunContext.llm_span`, so swapping out the backend
does not require touching pipeline code.
"""

from __future__ import annotations

import os

from loguru import logger


def init_phoenix(run_id: str, project_name: str | None = None):
    """Initialize Phoenix tracing for a run.

    Returns an OpenTelemetry `Tracer`. Auto-instrumentation is disabled so we
    get explicit control over the describe / reason / verify span tree;
    HTTP-level auto-instrumentation would flatten the interesting hierarchy.

    Safe to call once per run. If Phoenix cannot be initialized (e.g. the
    collector endpoint is unreachable or opentelemetry is not installed) this
    falls back to a no-op tracer so runs still complete and the JSONL source
    of truth is unaffected.
    """
    project = project_name or os.environ.get("PHOENIX_PROJECT_NAME", "tts-small-vlm")
    try:
        from opentelemetry import trace
        from phoenix.otel import register

        tracer_provider = register(
            project_name=project,
            auto_instrument=False,
        )
        tracer = trace.get_tracer("tts-small-vlm", tracer_provider=tracer_provider)
        logger.info("Phoenix tracing enabled (project={}, run_id={})", project, run_id)
        return tracer
    except Exception as exc:
        logger.warning(
            "Phoenix init failed ({}); falling back to no-op tracer. "
            "JSONL output is unaffected.",
            exc,
        )
        try:
            from opentelemetry import trace
            return trace.get_tracer("tts-small-vlm-noop")
        except ImportError:
            return _NoopTracer()


class _NoopSpan:
    """Minimal stand-in when opentelemetry is not installed."""

    def set_attribute(self, key, value):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoopTracer:
    """Minimal stand-in when opentelemetry is not installed."""

    def start_as_current_span(self, name, **kwargs):
        return _NoopSpan()
