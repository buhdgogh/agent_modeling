"""Durable tasks, cooperative cancellation, concurrency limits and call accounting."""
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from datetime import datetime, timezone
import hashlib
import math
import json
import os
from pathlib import Path
import sqlite3
import shutil
import threading
import time
import uuid

from agent_modeling.config import RUNTIME_ROOT
from agent_modeling.models import ArtifactRecord, CallEvent, TaskRecord


class ExecutionCancelled(BaseException):
    """Bypass specialist fallback/retry handlers when the caller cancels a task."""


def plain(value):
    if type(value).__name__ == "Overwrite":
        return plain(value.value)
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()
                if k not in {"messages", "chat_history"} and not str(k).endswith("base64")}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot persist task value: {type(value).__name__}")


class TaskStore:
    def __init__(self, path=None):
        self.path = str(path or RUNTIME_ROOT / "checkpoints" / "tasks.sqlite")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
                       "status TEXT NOT NULL, version INTEGER NOT NULL, cancel_requested INTEGER DEFAULT 0, "
                       "updated_at REAL NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                       "event_id TEXT UNIQUE NOT NULL, task_id TEXT NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS task_events ON events(task_id,seq)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, task_id, session_id):
        record = TaskRecord(task_id=task_id, session_id=str(session_id), status="running").model_dump()
        with self.connection() as db:
            db.execute("INSERT INTO tasks(task_id,session_id,status,version,updated_at,payload) VALUES(?,?,?,?,?,?)",
                       (task_id, str(session_id), "running", 1, time.time(), json.dumps(record)))
        return record

    def get(self, task_id, session_id=None):
        with self.connection() as db:
            row = db.execute("SELECT session_id,status,version,payload FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if not row or (session_id is not None and row[0] != str(session_id)):
            raise ValueError("任务不存在或不属于当前会话")
        payload = json.loads(row[3])
        payload.update(status=row[1], version=row[2])
        return payload

    def save(self, task_id, payload):
        TaskRecord.model_validate(payload)
        with self.connection() as db:
            changed = db.execute("UPDATE tasks SET status=?,updated_at=?,payload=? WHERE task_id=? AND version=?",
                                 (payload["status"], time.time(), json.dumps(plain(payload), ensure_ascii=False, allow_nan=False),
                                  task_id, payload["version"])).rowcount
            if changed != 1:
                raise ValueError("任务版本已变化，拒绝覆盖旧版本")

    def claim(self, task_id, session_id, expected_version):
        with self.connection() as db:
            changed = db.execute(
                "UPDATE tasks SET status='running',version=version+1,cancel_requested=0,updated_at=? "
                "WHERE task_id=? AND session_id=? AND version=? AND status IN ('awaiting_review','cancelled','failed')",
                (time.time(), task_id, str(session_id), expected_version)).rowcount
        if changed != 1:
            raise ValueError("任务正在执行、已完成或版本已变化，请刷新后重试")
        return self.get(task_id, session_id)

    def list(self, session_id):
        with self.connection() as db:
            rows = db.execute("SELECT task_id FROM tasks WHERE session_id=? ORDER BY updated_at DESC", (str(session_id),)).fetchall()
        records = []
        for row in rows:
            record = self.get(row[0], session_id)
            record.pop("snapshot", None)
            records.append(record)
        return records

    def delete_session(self, session_id):
        with self.connection() as db:
            db.execute("DELETE FROM events WHERE task_id IN (SELECT task_id FROM tasks WHERE session_id=?)", (str(session_id),))
            db.execute("DELETE FROM tasks WHERE session_id=?", (str(session_id),))

    def cancel(self, task_id, session_id):
        with self.connection() as db:
            return db.execute("UPDATE tasks SET cancel_requested=1 WHERE task_id=? AND session_id=? AND status='running'",
                              (task_id, str(session_id))).rowcount == 1

    def cancelled(self, task_id):
        with self.connection() as db:
            row = db.execute("SELECT cancel_requested FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return bool(row and row[0])

    def append(self, event):
        event = CallEvent.model_validate(event).model_dump()
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO events(event_id,task_id,payload) VALUES(?,?,?)",
                       (event["event_id"], event["task_id"], json.dumps(event, ensure_ascii=False)))

    def events(self, task_id):
        with self.connection() as db:
            rows = db.execute("SELECT payload FROM events WHERE task_id=? ORDER BY seq", (task_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]


class ExecutionController:
    """One process-wide limit, shared by parent nodes and all agent worker pools."""
    def __init__(self, max_model_calls=4, max_workers=4, max_nodes=2):
        if min(max_model_calls, max_workers, max_nodes) < 1:
            raise ValueError("Concurrency limits must be positive")
        self.max_workers, self.max_nodes = max_workers, max_nodes
        self.model_slots = threading.BoundedSemaphore(max_model_calls)
        self.tool_slots = threading.BoundedSemaphore(max_workers)
        self.task_slots = threading.BoundedSemaphore(max_workers)
        self._agent_locks = {}
        self._lock = threading.Lock()

    def agent_lock(self, agent):
        with self._lock:
            return self._agent_locks.setdefault(id(agent), threading.Lock())


_controller = None
_controller_lock = threading.Lock()
_current = ContextVar("agent_execution", default=None)
_parent_call = ContextVar("agent_parent_call", default="")


def controller():
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = ExecutionController(
                int(os.getenv("APP_MAX_MODEL_CONCURRENCY", "4")),
                int(os.getenv("APP_MAX_WORKERS", "4")), int(os.getenv("APP_MAX_NODE_CONCURRENCY", "2")))
    return _controller


def current_execution():
    return _current.get()


class ExecutionContext:
    def __init__(self, store, task_id, version, controls=None):
        self.store, self.task_id, self.version = store, task_id, version
        self.controls = controls or controller()
        self.started = time.perf_counter()
        try:
            self.prices = json.loads(os.getenv("APP_MODEL_PRICES_JSON", "{}"))
            if not isinstance(self.prices, dict):
                self.prices = {}
        except ValueError:
            self.prices = {}
        self._stop = threading.Event()

    def check(self):
        if self._stop.is_set() or self.store.cancelled(self.task_id) or (self.task_id, self.version) in _cancelled_jobs:
            self._stop.set()
            raise ExecutionCancelled()

    @contextmanager
    def bind(self):
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    @contextmanager
    def acquire(self, semaphore):
        while not semaphore.acquire(timeout=0.1):
            self.check()
        try:
            self.check()
            yield
        finally:
            semaphore.release()

    @contextmanager
    def call(self, kind, name, model="", attempt=1, operation_id=""):
        self.check()
        call_id, started = uuid.uuid4().hex, time.perf_counter()
        event = dict(call_id=call_id, kind=kind, name=name, model=model, attempt=attempt,
                     parent_call_id=_parent_call.get(), operation_id=operation_id or call_id,
                     input_tokens=None, output_tokens=None, cost=None)
        self.record({**event, "status": "started"})
        token = _parent_call.set(call_id)
        try:
            yield event
            self.check()
        except BaseException as exc:
            interrupted = type(exc).__name__ == "GraphInterrupt"
            event.update(status="interrupted" if interrupted else "cancelled" if isinstance(exc, ExecutionCancelled) else "failed",
                          error_code="" if interrupted else str(getattr(exc, "code", None) or type(exc).__name__))
            raise
        else:
            event.setdefault("status", "success")
        finally:
            _parent_call.reset(token)
            self.record({**event, "duration_ms": round((time.perf_counter() - started) * 1000, 2)})

    def record(self, event):
        self.store.append({"event_id": uuid.uuid4().hex, "task_id": self.task_id, "version": self.version,
                           "timestamp": datetime.now(timezone.utc).isoformat(), **event})

    def usage(self, event, response):
        usage = getattr(response, "usage_metadata", None) or {}
        metadata = getattr(response, "response_metadata", None) or {}
        usage = usage or metadata.get("token_usage") or metadata.get("usage") or {}
        incoming = usage.get("input_tokens", usage.get("prompt_tokens"))
        outgoing = usage.get("output_tokens", usage.get("completion_tokens"))
        if incoming is not None and outgoing is not None:
            try:
                if isinstance(incoming, bool) or isinstance(outgoing, bool) or min(int(incoming), int(outgoing)) < 0:
                    return
                incoming, outgoing = int(incoming), int(outgoing)
            except (ValueError, TypeError, OverflowError):
                return
            event.update(input_tokens=incoming, output_tokens=outgoing)
            price = self.prices.get(event["model"])
            if isinstance(price, dict) and all(isinstance(price.get(key), (int, float)) and
                    not isinstance(price[key], bool) and math.isfinite(price[key]) and price[key] >= 0
                    for key in ("input_per_million", "output_per_million")) and isinstance(price.get("currency", "CNY"), str):
                event["currency"] = price.get("currency", "CNY")
                event.update(input_price_per_million=price["input_per_million"], output_price_per_million=price["output_per_million"])
                event["cost"] = (int(incoming) * price["input_per_million"] +
                                 int(outgoing) * price["output_per_million"]) / 1_000_000


def check_cancelled():
    context = current_execution()
    if context:
        context.check()


def worker_count(default=4):
    context = current_execution()
    return min(default, (context.controls if context else controller()).max_workers)


def submit(executor, function, *args):
    return executor.submit(copy_context().run, function, *args)


def invoke_model(model, inputs, *, name="model", model_name=""):
    operation_id = uuid.uuid4().hex
    for attempt in range(1, 4):
        try:
            return _invoke_model_once(model, inputs, name=name, model_name=model_name, attempt=attempt, operation_id=operation_id)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            transient = isinstance(exc, (TimeoutError, ConnectionError)) or status in {408, 429, 500, 502, 503, 504}
            if not transient or attempt == 3:
                raise
            for _ in range(attempt * 5):
                check_cancelled()
                time.sleep(0.1)


def _invoke_model_once(model, inputs, *, name, model_name, attempt, operation_id):
    context = current_execution()
    if context is None:
        with controller().model_slots:
            return model.invoke(inputs)
    with context.acquire(context.controls.model_slots):
        with context.call("model", name, model_name or str(getattr(model, "model_name", "")),
                          attempt=attempt, operation_id=operation_id) as event:
            # Callbacks also observe raw chat results behind structured-output parsers.
            from langchain_core.callbacks import BaseCallbackHandler

            class UsageCallback(BaseCallbackHandler):
                def on_llm_end(self, response, **kwargs):
                    generations = getattr(response, "generations", [])
                    if generations and generations[0]:
                        message = getattr(generations[0][0], "message", None)
                        if message is not None:
                            context.usage(event, message)
                    output = getattr(response, "llm_output", None) or {}
                    if event["input_tokens"] is None:
                        from types import SimpleNamespace
                        context.usage(event, SimpleNamespace(response_metadata=output))

            response = model.invoke(inputs, config={"callbacks": [UsageCallback()]})
            if event["input_tokens"] is None:
                context.usage(event, response)
            return response


def invoke_tool(tool, arguments, name):
    context = current_execution()
    if context is None:
        with controller().tool_slots:
            return tool.invoke(arguments)
    with context.acquire(context.controls.tool_slots), context.call("tool", name):
        return tool.invoke(arguments)


def summarize_events(events):
    finished = [event for event in events if event["status"] != "started"]
    models = [event for event in finished if event["kind"] == "model"]
    runs = [event for event in finished if event["kind"] == "run"]
    priced = [event for event in models if event.get("cost") is not None]
    currencies = {event["currency"] for event in priced}
    complete = len(priced) == len(models) and len(currencies) <= 1
    return {"model_calls": len(models), "node_calls": sum(e["kind"] == "node" for e in finished),
            "cost_scope": "model_token_estimate",
            "tool_calls": sum(e["kind"] == "tool" for e in finished),
            "wall_duration_ms": round(sum(e["duration_ms"] for e in runs), 2),
            "model_duration_ms": round(sum(e["duration_ms"] for e in models), 2),
            "input_tokens": sum(e.get("input_tokens") or 0 for e in models),
            "output_tokens": sum(e.get("output_tokens") or 0 for e in models),
            "usage_unknown_calls": sum(e.get("input_tokens") is None for e in models),
            "cost_unknown_calls": len(models) - len(priced),
            "known_cost_by_currency": {currency: sum(e["cost"] for e in priced if e["currency"] == currency)
                                       for currency in currencies},
            "total_cost": sum(e["cost"] for e in priced) if complete else None,
            "currency": next(iter(currencies), None)}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            check_cancelled()
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path, task_id, version, kind, evidence_ids=()):
    path = Path(path)
    return ArtifactRecord(artifact_id=uuid.uuid4().hex, task_id=task_id, version=version, kind=kind,
                          path=str(path), sha256=file_sha256(path),
                          evidence_ids=list(evidence_ids)).model_dump()


def snapshot_inputs(state):
    """Pin file bytes before any model reads them; uploads may later be replaced."""
    root = Path(state["workspace_dir"]) / "sources"
    inputs = [p.strip() for p in (state.get("file_path") or "").split("|") if p.strip()]
    inputs += list(state.get("document_paths") or [])
    if state.get("image_path"):
        inputs.append(state["image_path"])
    for path in list(inputs):
        source = Path(path)
        if source.suffix.lower() == ".shp" and source.is_file():
            inputs.extend(str(p) for p in source.parent.glob(source.stem + ".*")
                          if p.suffix.lower() in {".shx", ".dbf", ".prj", ".cpg"})
        if source.name.lower() == "regions_ui.json":
            inputs.extend(str(source.parent / name) for name in ("meta.json", "legend_info.json")
                          if (source.parent / name).is_file())
    mapping, manifest = {}, []
    for path in dict.fromkeys(inputs):
        check_cancelled()
        source = Path(path)
        if not source.is_file():
            continue  # Let the owning parser/agent report a typed missing-file error.
        group = hashlib.sha256(str(source.resolve().parent).encode("utf-8")).hexdigest()[:16]
        target = root / group / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = file_sha256(target)
        mapping[path] = str(target)
        manifest.append({"original_path": path, "path": str(target), "sha256": digest})
    return {"file_path": "|".join(mapping.get(p.strip(), p.strip()) for p in (state.get("file_path") or "").split("|")),
            "image_path": mapping.get(state.get("image_path"), state.get("image_path", "")),
            "document_paths": [mapping.get(p, p) for p in state.get("document_paths") or []],
            "source_files": manifest, "sources_pinned": True}


_jobs, _jobs_lock, _pool = {}, threading.Lock(), None
_cancelled_jobs = set()


def launch_job(job_id, function, **kwargs):
    """Workers never call Streamlit; the UI polls their durable task/event state."""
    from concurrent.futures import ThreadPoolExecutor
    global _pool
    with _jobs_lock:
        if len(_jobs) > 64:
            for old_id, job in list(_jobs.items()):
                if job.done() and old_id != job_id:
                    _jobs.pop(old_id, None)
                    _clear_job_flags(old_id)
        if job_id in _jobs and not _jobs[job_id].done():
            raise ValueError("任务正在执行")
        if sum(not job.done() for job in _jobs.values()) >= 32:
            raise ValueError("等待执行的任务过多，请稍后重试")
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=controller().max_workers, thread_name_prefix="geology-task")
        _clear_job_flags(job_id)
        _jobs[job_id] = _pool.submit(function, **kwargs)
        _jobs[job_id].task_session_id = str(kwargs.get("session_id", "default"))
        _jobs[job_id].task_version = int(kwargs.get("expected_version", 0)) + 1
    return job_id


def get_job(task_id):
    with _jobs_lock:
        return _jobs.get(task_id)


def forget_job(task_id):
    with _jobs_lock:
        job = _jobs.get(task_id)
        if job is not None and job.done():
            _jobs.pop(task_id, None)
            _clear_job_flags(task_id)


def _clear_job_flags(task_id):
    _cancelled_jobs.difference_update({entry for entry in _cancelled_jobs if entry[0] == task_id})


def cancel_job(task_id, session_id):
    with _jobs_lock:
        job = _jobs.get(task_id)
        if job is None or job.done() or job.task_session_id != str(session_id):
            return False
        _cancelled_jobs.add((task_id, job.task_version))
        return True
