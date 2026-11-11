# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for packing a ModelBatch into one buffer."""

import copy

import torch

from weathergen.datasets.batch import ModelBatch, SampleMetaData
from weathergen.datasets.stream_data import StreamData

STREAMS = ["era5", "synop", "empty"]
INPUT_STEPS = 2
OUTPUT_STEPS = 3
CELLS = 12


def _stream_data(rng: torch.Generator, with_source: bool) -> StreamData:
    sd = StreamData(0, INPUT_STEPS, OUTPUT_STEPS, CELLS)
    for step in range(OUTPUT_STEPS):
        n = int(torch.randint(1, 50, (1,), generator=rng))
        sd.target_coords[step] = torch.rand(n, 7, generator=rng)
        sd.target_coords_lens[step] = torch.randint(0, 9, (CELLS,), generator=rng)
        sd.target_tokens[step] = torch.rand(n, 5, generator=rng, dtype=torch.float64)
    # one step without targets, as in the initial state of StreamData
    sd.target_tokens[OUTPUT_STEPS - 1] = torch.tensor([])
    if with_source:
        for step in range(INPUT_STEPS):
            n = int(torch.randint(1, 50, (1,), generator=rng))
            sd.source_tokens_cells[step] = torch.rand(n, 3, 4, generator=rng)
            sd.source_tokens_lens[step] = torch.randint(
                0, 9, (CELLS,), generator=rng, dtype=torch.int32
            )
            sd.source_idxs_embed[step] = torch.randint(0, 99, (n,), generator=rng)
    return sd


def _batch(seed: int = 0) -> ModelBatch:
    rng = torch.Generator().manual_seed(seed)
    batch = ModelBatch(STREAMS, 2, 1, 0, OUTPUT_STEPS)
    for samples, with_source in ((batch.source_samples, True), (batch.target_samples, False)):
        for sample in samples.samples:
            for name in STREAMS[:2]:
                sample.add_stream_data(name, _stream_data(rng, with_source))
                mask = torch.rand(CELLS, generator=rng) > 0.5
                sample.add_meta_info(name, SampleMetaData(params={}, mask=mask))
            # stream without mask; the stream "empty" has no data at all
            sample.add_meta_info("empty", SampleMetaData(params={}))
        samples.tokens_lens = torch.randint(0, 9, (INPUT_STEPS, len(samples), CELLS), generator=rng)
    return batch


def _slots(batch: ModelBatch) -> list:
    return batch.source_samples.device_slots() + batch.target_samples.device_slots()


def _tensors(batch: ModelBatch) -> list:
    return [container[key] for container, key in _slots(batch)]


def _assert_same_tensors(batch: ModelBatch, reference: ModelBatch) -> None:
    tensors, expected = _tensors(batch), _tensors(reference)
    assert len(tensors) == len(expected)
    for tensor, exp in zip(tensors, expected, strict=True):
        if exp is None:
            assert tensor is None
            continue
        assert tensor.dtype == exp.dtype
        assert tensor.shape == exp.shape
        assert torch.equal(tensor, exp)


def _num_storages(batch: ModelBatch) -> int:
    tensors = [t for t in _tensors(batch) if t is not None and t.numel() > 0]
    return len({t.untyped_storage().data_ptr() for t in tensors})


def test_pack_keeps_values_and_uses_one_buffer():
    reference = _batch()
    batch = _batch().pack()

    _assert_same_tensors(batch, reference)
    assert _num_storages(reference) > 30
    assert _num_storages(batch) == 1
    assert batch.packed_buffer.is_shared()
    assert batch.packed_buffer.untyped_storage().data_ptr() == next(
        t.untyped_storage().data_ptr() for t in _tensors(batch) if t is not None and t.numel() > 0
    )


def test_to_device_of_packed_batch_matches_unpacked():
    reference = _batch().to_device("cpu")
    batch = _batch().pack().to_device("cpu")

    _assert_same_tensors(batch, reference)
    assert batch.packed_buffer is None
    assert batch.device == "cpu"
    assert batch.source_samples.device == "cpu"
    assert batch.target_samples.device == "cpu"


def test_to_device_moves_packed_buffer_once():
    batch = _batch().pack()
    moved = []

    def move(tensor):
        moved.append(tensor)
        return tensor.clone()

    new_buffer = batch._move_packed(move)

    # only the buffer carries data; empty tensors are moved as before
    assert moved[0] is batch.packed_buffer
    assert all(t.numel() == 0 for t in moved[1:])
    _assert_same_tensors(batch, _batch())
    tensors = [t for t in _tensors(batch) if t is not None and t.numel() > 0]
    assert all(
        t.untyped_storage().data_ptr() == new_buffer.untyped_storage().data_ptr() for t in tensors
    )


def test_tensors_replaced_after_pack_are_moved_separately():
    reference = _batch()
    batch = _batch().pack()
    replaced = torch.arange(5.0)
    batch.source_samples.samples[0].streams_data["era5"].target_coords[0] = replaced
    reference.source_samples.samples[0].streams_data["era5"].target_coords[0] = replaced.clone()

    batch.to_device("cpu")

    _assert_same_tensors(batch, reference)


def test_deepcopy_of_packed_batch_can_be_moved():
    batch = copy.deepcopy(_batch().pack())

    batch.to_device("cpu")

    _assert_same_tensors(batch, _batch())


class _Dataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        for seed in range(2):
            yield _batch(seed).pack()


def test_packed_batch_from_data_loader_worker():
    loader = torch.utils.data.DataLoader(_Dataset(), batch_size=None, num_workers=1)

    for seed, batch in enumerate(loader):
        # the views still share the buffer after the transfer between processes
        assert _num_storages(batch) == 1
        batch.to_device("cpu")
        _assert_same_tensors(batch, _batch(seed))
