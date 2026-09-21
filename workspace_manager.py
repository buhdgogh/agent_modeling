"""Session-scoped runtime workspace management."""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
SESSIONS_ROOT = RUNTIME_ROOT / "sessions"
CHECKPOINT_ROOT = RUNTIME_ROOT / "checkpoints"


def _safe_segment(value: Any) -> str:
    segment = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "default"))
    segment = segment.strip(".")
    return segment[:96] or "default"


def safe_upload_name(filename: str) -> str:
    """Return a filesystem-safe basename for an uploaded file."""
    basename = Path(str(filename or "upload")).name
    safe_name = re.sub(r"[^\w.()\-\u4e00-\u9fff]", "_", basename, flags=re.UNICODE)
    return safe_name[:180] or "upload"


def session_dir(session_id: Any) -> Path:
    path = SESSIONS_ROOT / _safe_segment(session_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def upload_batch_dir(session_id: Any, batch_id: Any) -> Path:
    path = session_dir(session_id) / "uploads" / _safe_segment(batch_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_uploaded_file(uploaded_file: Any, session_id: Any, batch_id: Any) -> str:
    """Persist one Streamlit upload inside its session and upload batch."""
    filename = safe_upload_name(uploaded_file.name)
    extension = Path(filename).suffix.lower()
    base_dir = upload_batch_dir(session_id, batch_id)

    if extension in {".shp", ".shx", ".dbf", ".prj", ".cpg"}:
        target_dir = base_dir / _safe_segment(Path(filename).stem)
    else:
        target_dir = base_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    target = (target_dir / filename).resolve()
    if base_dir.resolve() not in target.parents:
        raise ValueError("上传文件路径超出当前会话工作区。")
    target.write_bytes(uploaded_file.getbuffer())
    return str(target)


def create_run_dir(session_id: Any, request_id: str | None = None) -> str:
    run_id = _safe_segment(request_id or uuid.uuid4().hex)
    path = session_dir(session_id) / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return str(path)


def checkpoint_db_path() -> str:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    return str((CHECKPOINT_ROOT / "langgraph.sqlite").resolve())


def clear_session_workspace(session_id: Any) -> None:
    """Remove one session's runtime files without touching other sessions."""
    root = SESSIONS_ROOT.resolve()
    target = (root / _safe_segment(session_id)).resolve()
    if target.parent != root:
        raise ValueError("拒绝清理会话目录之外的路径。")
    if target.exists():
        shutil.rmtree(target)
