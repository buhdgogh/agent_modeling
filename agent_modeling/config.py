"""Project paths, configuration and tracing."""


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / "api_key.env"
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
SESSIONS_ROOT = RUNTIME_ROOT / "sessions"
CHECKPOINT_ROOT = RUNTIME_ROOT / "checkpoints"
DEFAULT_IMAGE_OUTPUT = RUNTIME_ROOT / "standalone" / "image_analysis"
DEFAULT_BOREHOLE_OUTPUT = RUNTIME_ROOT / "standalone" / "boreholes"
PEACE_ROOT = PROJECT_ROOT / "PEACE"


def load_settings():
    from dotenv import load_dotenv

    return load_dotenv(ENV_FILE, override=True)


import os


def configure_tracing():
    enabled = os.getenv("APP_ENABLE_LANGSMITH", "false").strip().lower() in {"true", "1", "yes"}
    value = "true" if enabled else "false"
    # Override legacy flags as well, including flags loaded from api_key.env.
    os.environ["LANGSMITH_TRACING"] = value
    os.environ["LANGCHAIN_TRACING_V2"] = value
    return enabled
