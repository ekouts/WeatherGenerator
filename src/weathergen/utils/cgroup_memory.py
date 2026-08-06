# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Diagnostic cgroup-v2 host-memory timeline collection."""

import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "diagnostic.cgroup_memory"
_REQUIRED_STAT_FIELDS = ("anon", "file", "shmem")
_REQUIRED_EVENT_FIELDS = ("high", "max", "oom", "oom_kill")


def resolve_cgroup_v2_path(
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_mount: Path = Path("/sys/fs/cgroup"),
    slurm_job_id: str | None = None,
) -> Path:
    """Resolve the cgroup-v2 directory that best represents the current Slurm job."""
    unified_path: str | None = None
    for line in proc_cgroup_path.read_text().splitlines():
        hierarchy_id, controllers, relative_path = line.split(":", maxsplit=2)
        if hierarchy_id == "0" and controllers == "":
            unified_path = relative_path
            break

    if unified_path is None:
        raise RuntimeError("The current process does not expose a cgroup-v2 hierarchy")

    relative_path = unified_path.lstrip("/")
    current_path = cgroup_mount / relative_path
    if not (current_path / "memory.current").is_file():
        # A cgroup namespace can expose its root as '/', while the mount itself is
        # already rooted at the process's delegated cgroup.
        if relative_path or not (cgroup_mount / "memory.current").is_file():
            raise RuntimeError(f"Cannot read cgroup-v2 memory metrics from {current_path}")
        current_path = cgroup_mount

    job_id = slurm_job_id if slurm_job_id is not None else os.environ.get("SLURM_JOB_ID")
    if not job_id or current_path == cgroup_mount:
        return current_path

    job_dir_names = {f"job_{job_id}", f"job-{job_id}.scope"}
    for candidate in (current_path, *current_path.parents):
        if candidate == cgroup_mount.parent:
            break
        if candidate.name in job_dir_names and (candidate / "memory.current").is_file():
            return candidate
        if candidate == cgroup_mount:
            break

    logger.warning(
        "Could not identify a job-level cgroup for SLURM_JOB_ID=%s; sampling %s",
        job_id,
        current_path,
    )
    return current_path


def _read_key_value_file(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, value = line.split(maxsplit=1)
        values[key] = int(value)
    return values


def read_cgroup_memory_snapshot(cgroup_path: Path) -> dict[str, float]:
    """Read the small set of cgroup-v2 counters used by the diagnostic timeline."""
    memory_stat = _read_key_value_file(cgroup_path / "memory.stat")
    memory_events = _read_key_value_file(cgroup_path / "memory.events")

    missing_stat = set(_REQUIRED_STAT_FIELDS) - memory_stat.keys()
    missing_events = set(_REQUIRED_EVENT_FIELDS) - memory_events.keys()
    if missing_stat or missing_events:
        raise RuntimeError(
            f"Incomplete cgroup memory counters: stat={sorted(missing_stat)}, "
            f"events={sorted(missing_events)}"
        )

    snapshot = {
        f"{_METRIC_PREFIX}.current_bytes": float(
            (cgroup_path / "memory.current").read_text().strip()
        ),
        f"{_METRIC_PREFIX}.anon_bytes": float(memory_stat["anon"]),
        f"{_METRIC_PREFIX}.file_bytes": float(memory_stat["file"]),
        f"{_METRIC_PREFIX}.shmem_bytes": float(memory_stat["shmem"]),
        f"{_METRIC_PREFIX}.events.high": float(memory_events["high"]),
        f"{_METRIC_PREFIX}.events.max": float(memory_events["max"]),
        f"{_METRIC_PREFIX}.events.oom": float(memory_events["oom"]),
        f"{_METRIC_PREFIX}.events.oom_kill": float(memory_events["oom_kill"]),
    }
    peak_path = cgroup_path / "memory.peak"
    if peak_path.is_file():
        snapshot[f"{_METRIC_PREFIX}.peak_bytes"] = float(peak_path.read_text().strip())
    return snapshot


class CgroupMemoryTimeline:
    """Sample job host memory and asynchronously log trainer stage markers."""

    def __init__(
        self,
        log_fn: Callable[[dict[str, float]], None],
        sampling_interval_ms: int = 100,
        cgroup_path: Path | None = None,
    ) -> None:
        if sampling_interval_ms <= 0:
            raise ValueError("sampling_interval_ms must be greater than zero")

        self._log_fn = log_fn
        self._sampling_interval_s = sampling_interval_ms / 1_000
        self._cgroup_path = cgroup_path or resolve_cgroup_v2_path()
        self._pending_events: queue.SimpleQueue[dict[str, float]] = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Validate the source before starting a background thread, so configuration
        # errors fail visibly in the trainer process.
        read_cgroup_memory_snapshot(self._cgroup_path)

    @property
    def cgroup_path(self) -> Path:
        return self._cgroup_path

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="cgroup-memory-timeline",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, 2 * self._sampling_interval_s))
        self._thread = None

    def record_stage(
        self,
        name: str,
        *,
        monotonic_ns: int | None = None,
        **values: int | float,
    ) -> None:
        """Queue a timestamped marker without performing file I/O in the training path."""
        marker = {
            "diagnostic.timeline.monotonic_ns": float(
                time.monotonic_ns() if monotonic_ns is None else monotonic_ns
            ),
            f"diagnostic.timeline.stage.{name}": 1.0,
        }
        marker.update({f"diagnostic.timeline.{key}": float(value) for key, value in values.items()})
        self._pending_events.put(marker)

    def _run(self) -> None:
        try:
            while True:
                sample = read_cgroup_memory_snapshot(self._cgroup_path)
                sample["diagnostic.timeline.monotonic_ns"] = float(time.monotonic_ns())
                sample["diagnostic.timeline.sample"] = 1.0
                self._log_fn(sample)
                self._drain_pending_events()
                if self._stop_event.is_set():
                    break
                self._stop_event.wait(self._sampling_interval_s)
            self._drain_pending_events()
        except Exception:
            logger.exception("Cgroup memory timeline stopped after an unexpected error")

    def _drain_pending_events(self) -> None:
        while True:
            try:
                marker = self._pending_events.get_nowait()
            except queue.Empty:
                return
            self._log_fn(marker)
