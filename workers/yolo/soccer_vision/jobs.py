"""In-memory job registry for long-running video analysis.

/analyze-video returns a job ID immediately; the dense pass runs on a worker
thread and reports stage + frame progress here. The frontend polls
/jobs/{id} so a stall is visible (updatedAt stops advancing) instead of an
indefinite spinner, and errors carry the actual exception message.

In-memory only: jobs do not survive a worker restart — a poll for an unknown
job ID returns 404 and the client should treat that as "re-upload".
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import logger

# Completed jobs are kept for this long so the client can fetch the result,
# then reaped to bound memory (dense frames for a long clip are large).
JOB_TTL_SECONDS = 15 * 60


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"   # queued | calibrating | tracking | postprocessing | done | error
    frames_done: int = 0
    frames_total: int = 0
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.updated_at = time.time()

    def to_status_dict(self) -> dict[str, Any]:
        percent = None
        if self.frames_total > 0:
            percent = round(100.0 * self.frames_done / self.frames_total, 1)
        return {
            "jobId": self.id,
            "status": self.status,
            "stage": self.stage,
            "framesDone": self.frames_done,
            "framesTotal": self.frames_total,
            "percent": percent,
            "error": self.error,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "elapsedSeconds": round((self.finished_at or time.time()) - self.created_at, 1),
        }


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create_job() -> Job:
    _reap()
    job = Job(id=uuid.uuid4().hex[:12])
    with _lock:
        _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def _reap() -> None:
    now = time.time()
    with _lock:
        stale = [
            jid for jid, job in _jobs.items()
            if job.finished_at is not None and now - job.finished_at > JOB_TTL_SECONDS
        ]
        for jid in stale:
            logger.info("reaping finished job %s", jid)
            del _jobs[jid]
