"""
Background jobs for long tasks (preprocessing, training, SHAP, tuning).

A job runs in a worker thread so the page stays responsive; the page polls it with a
fragment that re-runs every second and shows a progress bar, the current message and a
Cancel button. The worker never calls Streamlit APIs: it only updates the Job object;
the page applies the result once the job has finished.
"""
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import streamlit as st


@dataclass
class Job:
    key: str
    label: str
    step: Optional[str] = None
    status: str = "running"            # running | done | error | cancelled
    progress: float = 0.0
    message: str = "Starting..."
    result: Any = None
    error: Optional[str] = None
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    applied: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def update(self, progress: float = None, message: str = None):
        if progress is not None:
            self.progress = max(0.0, min(1.0, float(progress)))
        if message is not None:
            self.message = message

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started


def jobs() -> dict:
    if "jobs" not in st.session_state:
        st.session_state.jobs = {}
    return st.session_state.jobs


def get_job(key: str) -> Optional[Job]:
    return jobs().get(key)


def start_job(key: str, label: str, fn: Callable[[Job], Any], step: str = None) -> Job:
    """Run ``fn(job)`` in a daemon thread; its return value becomes ``job.result``."""
    job = Job(key=key, label=label, step=step)

    def run():
        try:
            job.result = fn(job)
            job.status = "cancelled" if job.cancel_event.is_set() and job.result is None else "done"
        except Exception as exc:  # shown in the UI with the traceback tail
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
        finally:
            job.finished = time.time()
            job.progress = 1.0 if job.status == "done" else job.progress

    jobs()[key] = job
    threading.Thread(target=run, name=f"job-{key}", daemon=True).start()
    return job


def render_running(job: Job):
    """Progress UI for a running job; re-runs every second and triggers a full rerun when it finishes."""

    @st.fragment(run_every=1.0)
    def _poll():
        if job.status != "running":
            st.rerun(scope="app")
        st.progress(job.progress, text=f"{job.label}: {job.message} ({job.elapsed:.0f} s)")
        if st.button("Cancel", key=f"cancel_{job.key}", icon=":material/stop_circle:",
                     disabled=job.cancel_event.is_set()):
            job.cancel_event.set()
            job.message = "Cancelling after the current unit of work..."

    _poll()
