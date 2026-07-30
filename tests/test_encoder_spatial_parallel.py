# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch

from weathergen.datasets.healpix_domain import build_local_healpix_cell_splits
from weathergen.utils import distributed


def test_local_healpix_construction_matches_global_cell_slices():
    num_cells = 48
    cell_ids = np.repeat(np.arange(num_cells), np.arange(num_cells) % 3 + 1)
    rng = np.random.default_rng(7)
    cell_ids = cell_ids[rng.permutation(len(cell_ids))]
    global_cells = build_local_healpix_cell_splits(
        cell_ids,
        num_cells,
        cell_start=0,
        cell_end=num_cells,
    )
    cells_per_rank = len(global_cells) // 4

    local_cells_all = []
    for spatial_rank in range(4):
        cell_start = spatial_rank * cells_per_rank
        cell_end = cell_start + cells_per_rank
        local_cells = build_local_healpix_cell_splits(
            cell_ids,
            num_cells,
            cell_start=cell_start,
            cell_end=cell_end,
        )

        assert len(local_cells) == cells_per_rank
        for local_cell, global_cell in zip(
            local_cells,
            global_cells[cell_start:cell_end],
            strict=True,
        ):
            np.testing.assert_array_equal(local_cell, global_cell)
        local_cells_all.extend(local_cells)

    assert len(local_cells_all) == len(global_cells)
    for local_cell, global_cell in zip(local_cells_all, global_cells, strict=True):
        np.testing.assert_array_equal(local_cell, global_cell)


def test_local_healpix_construction_rejects_invalid_range():
    with pytest.raises(ValueError, match="invalid HEALPix cell range"):
        build_local_healpix_cell_splits(
            np.arange(12),
            num_cells=12,
            cell_start=6,
            cell_end=13,
        )

def test_spatial_parallel_size_requires_whole_rank_groups(monkeypatch):
    monkeypatch.setattr(distributed, "get_world_size", lambda: 16)
    assert distributed.get_encoder_spatial_parallel_size({"encoder_spatial_parallel_size": 4}) == 4
    assert distributed.get_encoder_spatial_parallel_size({"encoder_spatial_parallel_size": 8}) == 8

    with pytest.raises(ValueError, match="must be divisible"):
        distributed.get_encoder_spatial_parallel_size({"encoder_spatial_parallel_size": 6})
