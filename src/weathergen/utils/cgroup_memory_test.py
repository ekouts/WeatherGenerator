# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

from weathergen.utils.cgroup_memory import (
    CgroupMemoryTimeline,
    read_cgroup_memory_snapshot,
    resolve_cgroup_v2_path,
)


def _write_cgroup_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "memory.current").write_text("1200\n")
    (path / "memory.peak").write_text("2400\n")
    (path / "memory.stat").write_text("anon 700\nfile 400\nshmem 100\n")
    (path / "memory.events").write_text("low 0\nhigh 1\nmax 2\noom 3\noom_kill 4\n")


def test_resolve_slurm_job_cgroup(tmp_path: Path) -> None:
    cgroup_mount = tmp_path / "cgroup"
    job_path = cgroup_mount / "slurm" / "job_42"
    task_path = job_path / "step_0" / "task_0"
    _write_cgroup_files(job_path)
    _write_cgroup_files(task_path)
    proc_cgroup = tmp_path / "proc-cgroup"
    proc_cgroup.write_text("0::/slurm/job_42/step_0/task_0\n")

    assert resolve_cgroup_v2_path(proc_cgroup, cgroup_mount, slurm_job_id="42") == job_path


def test_read_cgroup_memory_snapshot(tmp_path: Path) -> None:
    _write_cgroup_files(tmp_path)

    assert read_cgroup_memory_snapshot(tmp_path) == {
        "diagnostic.cgroup_memory.current_bytes": 1200.0,
        "diagnostic.cgroup_memory.peak_bytes": 2400.0,
        "diagnostic.cgroup_memory.anon_bytes": 700.0,
        "diagnostic.cgroup_memory.file_bytes": 400.0,
        "diagnostic.cgroup_memory.shmem_bytes": 100.0,
        "diagnostic.cgroup_memory.events.high": 1.0,
        "diagnostic.cgroup_memory.events.max": 2.0,
        "diagnostic.cgroup_memory.events.oom": 3.0,
        "diagnostic.cgroup_memory.events.oom_kill": 4.0,
    }


def test_timeline_records_samples_and_markers(tmp_path: Path) -> None:
    _write_cgroup_files(tmp_path)
    records: list[dict[str, float]] = []
    timeline = CgroupMemoryTimeline(records.append, sampling_interval_ms=10, cgroup_path=tmp_path)

    timeline.start()
    timeline.record_stage("batch_dequeued", monotonic_ns=123, batch_index=2)
    timeline.stop()

    assert any(record.get("diagnostic.timeline.sample") == 1.0 for record in records)
    marker = next(
        record
        for record in records
        if record.get("diagnostic.timeline.stage.batch_dequeued") == 1.0
    )
    assert marker["diagnostic.timeline.monotonic_ns"] == 123.0
    assert marker["diagnostic.timeline.batch_index"] == 2.0
