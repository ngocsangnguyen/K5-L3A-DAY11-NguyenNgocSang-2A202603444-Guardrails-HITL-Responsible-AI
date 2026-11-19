"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from agents.security_boundary import contains_secret, normalize_for_security
from assignment._artifacts import write_json
from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], tuple[float, dict]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store a sanitized input and start timestamp until the request finishes."""
        self._open[(user_id, request_id)] = (
            time.perf_counter(),
            {"input": _audit_text(text), "started_at": utc_now_iso()},
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        """Record the response, decision, and elapsed time."""
        opened = self._open.pop((user_id, request_id), None)
        latency_ms = None
        input_fields = {"input": None, "started_at": None}
        if opened is not None:
            started, input_fields = opened
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
        entry = {
            "request_id": request_id,
            "user_id": user_id,
            **input_fields,
            "output": _audit_text(text),
            "blocked": blocked,
            "layer": layer,
            "latency_ms": latency_ms,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        write_json(filepath or default_audit_log_path(), self.logs)


def _audit_text(text: str) -> str:
    filtered = content_filter(normalize_for_security(text))
    if not filtered["safe"]:
        text = filtered["redacted"]
    if contains_secret(text):
        return "[REDACTED]"
    return text


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
