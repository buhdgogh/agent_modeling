"""Explicit opt-in for external telemetry; local monitoring stays enabled."""

import os


def configure_tracing():
    enabled = os.getenv("APP_ENABLE_LANGSMITH", "false").strip().lower() in {"true", "1", "yes"}
    value = "true" if enabled else "false"
    # Override legacy flags as well, including flags loaded from api_key.env.
    os.environ["LANGSMITH_TRACING"] = value
    os.environ["LANGCHAIN_TRACING_V2"] = value
    return enabled
