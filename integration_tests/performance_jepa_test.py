# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Performance integration test for the Weather Generator.
Verifies that throughput metrics are produced during JEPA training and that
global samples/sec stays above a minimum threshold.

Must run on a GPU machine. For multi-GPU runs, launch via sbatch so that
MASTER_ADDR is set and torch.distributed initialises from SLURM env vars.
Only the task with SLURM_PROCID=0 asserts on the throughput threshold.

Command (single GPU):
    uv run pytest ./integration_tests/performance_jepa_test.py

Command (multi-GPU via SLURM, e.g. 4x GH200 on Santis):
    sbatch <your_sbatch_script>.sh
"""

import json
import logging
import os
import shutil
from pathlib import Path

import pytest

from weathergen.run_train import main
from weathergen.utils.metrics import get_train_metrics_path

logger = logging.getLogger(__name__)

try:
    from git import Repo

    repo = Repo(search_parent_directories=False)
    commit_hash = repo.head.object.hexsha[:5]
    logger.info(f"Current commit hash: {commit_hash}")
except Exception as e:
    commit_hash = "unknown"
    logger.warning(f"Could not get commit hash: {e}")

WEATHERGEN_HOME = Path(__file__).parent.parent

METRIC_NAME = "performance.throughput.global.samples_per_sec"
# TODO: decide on a reasonable threshold
THROUGHPUT_THRESHOLD = 100.0


@pytest.fixture()
def setup(test_run_id):
    logger.info(f"setup fixture with {test_run_id}")
    shutil.rmtree(WEATHERGEN_HOME / "results" / test_run_id, ignore_errors=True)
    shutil.rmtree(WEATHERGEN_HOME / "models" / test_run_id, ignore_errors=True)
    yield
    logger.info("end fixture")


@pytest.mark.parametrize("test_run_id", ["test_performance_jepa_" + commit_hash])
def test_throughput(setup, test_run_id):
    logger.info(f"test_throughput with run_id {test_run_id} {WEATHERGEN_HOME}")

    main(
        [
            "train",
            f"--config={WEATHERGEN_HOME}/integration_tests/jepa1.yaml",
            f"--config={WEATHERGEN_HOME}/integration_tests/performance_jepa.yaml",
            "--run-id",
            test_run_id,
        ]
    )

    # Use SLURM_PROCID rather than torch.distributed state: it is set before
    # any Python runs and remains valid even if the process group was never
    # initialised or was torn down by an error inside main().
    if os.environ.get("SLURM_PROCID", "0") == "0":
        assert_throughput_above_threshold(test_run_id)

    logger.info("end test_throughput")


def load_metrics(run_id):
    file_path = get_train_metrics_path(base_path=WEATHERGEN_HOME / "results" / run_id, run_id=run_id)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found for run_id: {run_id}")
    with open(file_path) as f:
        json_str = f.readlines()
    return json.loads("[" + "".join([s.replace("\n", ",") for s in json_str])[:-1] + "]")


def assert_throughput_above_threshold(run_id):
    metrics = load_metrics(run_id)
    # Take the last logged value — most stable, fully post-warmup.
    throughput = next(
        (
            metric.get(METRIC_NAME)
            for metric in reversed(metrics)
            if metric.get(METRIC_NAME) is not None
        ),
        None,
    )
    assert throughput is not None, (
        f"'{METRIC_NAME}' not found in metrics — is track_performance_metrics enabled?"
    )
    assert throughput > THROUGHPUT_THRESHOLD, (
        f"'{METRIC_NAME}' is {throughput:.2f} samples/sec, expected above {THROUGHPUT_THRESHOLD}"
    )
