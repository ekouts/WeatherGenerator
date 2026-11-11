"""
Data structures for student-teacher multi-view training.

Provides clean separation between:
  - Model data (StreamData objects containing tensors)
  - View metadata (spatial masks, strategies, relationships)
"""

import copy
import typing
from dataclasses import dataclass

import numpy as np
import torch

from weathergen.common.config import Config
from weathergen.datasets.stream_data import StreamData

# alignment of the tensors inside a packed buffer; covers every torch dtype (max. 16 bytes)
_PACK_ALIGNMENT = 16


def _is_packable(tensor) -> bool:
    """True for the tensors that ModelBatch.pack() moves into the packed buffer."""
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.device.type == "cpu"
        and tensor.layout == torch.strided
        and tensor.numel() > 0
        and not tensor.requires_grad
    )


@dataclass
class SampleMetaData:
    # sample parameters (masking)
    params: Config | dict

    mask: torch.Tensor | None = None

    global_params: dict | None = None

    def add_global_params(self, params: dict) -> None:
        if self.global_params is None:
            self.global_params = {}
        self.global_params.update(params)

    def add_params(self, params: dict) -> None:
        if self.params is None:
            self.params = {}
        self.params.update(params)


class Sample:
    # keys: stream name, values: SampleMetaData
    meta_info: dict[str | SampleMetaData]

    # stream-independent meta information of sample
    sample_meta_info: dict

    # data for all streams
    # keys: stream_name, values: StreamData
    streams_data: dict[str, StreamData | None]

    def pin_memory(self):
        """Pin all tensors in this Sample to CPU pinned memory"""

        # Pin StreamData objects in streams_data dict
        if hasattr(self, "streams_data") and isinstance(self.streams_data, dict):
            for _stream_name, stream_data in self.streams_data.items():
                if stream_data is not None and hasattr(stream_data, "pin_memory"):
                    stream_data.pin_memory()

        # Pin tensors in meta_info
        if hasattr(self, "meta_info") and isinstance(self.meta_info, dict):
            for _key, meta_data in self.meta_info.items():
                if isinstance(meta_data, SampleMetaData):
                    # Pin mask tensor
                    if meta_data.mask is not None and isinstance(meta_data.mask, torch.Tensor):
                        meta_data.mask = meta_data.mask.pin_memory()

        return self

    def __init__(self, stream_names: list[str]) -> None:
        self.meta_info = {}
        self.sample_meta_info = {}

        self.streams_data = {}
        for stream_name in stream_names:
            self.streams_data[stream_name] = None

    def to_device(
        self,
        device,
        target_steps: list[int] | None = None,
        include_source: bool = True,
        include_target_coords: bool = True,
        include_target_tokens: bool = True,
    ) -> None:
        for key in self.meta_info.keys():
            self.meta_info[key].mask = (
                self.meta_info[key].mask.to(device, non_blocking=True)
                if self.meta_info[key].mask is not None
                else None
            )

        for key, val in self.streams_data.items():
            if val is not None:
                self.streams_data[key] = val.to_device(
                    device,
                    target_steps,
                    include_source,
                    include_target_coords,
                    include_target_tokens,
                )

    def device_slots(self) -> list[tuple[list | dict, int | str]]:
        """
        Where the tensors that to_device() moves by default are stored, as (container, key)
        pairs: container[key] reads the tensor and can be assigned to replace it.
        """
        slots = [(meta_data.__dict__, "mask") for meta_data in self.meta_info.values()]
        for stream_data in self.streams_data.values():
            if stream_data is not None:
                slots.extend(stream_data.device_slots())
        return slots

    def clear_target_coordinates(self, target_steps: list[int]) -> None:
        for stream_data in self.streams_data.values():
            if stream_data is not None:
                stream_data.clear_target_coordinates(target_steps)

    def is_empty(self) -> bool:
        """
        Check if sample is empty
        """
        empty = [s.empty() if s is not None else True for _, s in self.streams_data.items()]
        return np.array(empty).all()

    def is_nan(self) -> bool:
        """
        Check if sample is all NaN
        """
        is_nan = [s.nan() if s is not None else False for _, s in self.streams_data.items()]
        return np.array(is_nan).all()

    def sources_empty(self) -> bool:
        """
        Check if sources for sample are empty
        """
        empty = [s.source_empty() if s is not None else True for _, s in self.streams_data.items()]
        return np.array(empty).all()

    def sources_nan(self) -> bool:
        """
        Check if sources for sample are all NaN
        """
        is_nan = [s.source_nan() if s is not None else False for _, s in self.streams_data.items()]
        return np.array(is_nan).all()

    def targets_empty(self) -> bool:
        """
        Check if targets for sample are empty
        """
        empty = [s.target_empty() if s is not None else True for _, s in self.streams_data.items()]
        return np.array(empty).all()

    def targets_nan(self) -> bool:
        """
        Check if targets for sample are all NaN
        """
        is_nan = [s.target_nan() if s is not None else False for _, s in self.streams_data.items()]
        return np.array(is_nan).all()

    def add_stream_data(self, stream_name: str, stream_data: StreamData) -> None:
        """
        Add data for stream @stream_name to sample
        """
        assert self.streams_data.get(stream_name, -1) != -1, "stream name does not exist"
        self.streams_data[stream_name] = stream_data

    def add_meta_info(self, stream_name: str, meta_info: SampleMetaData) -> None:
        """
        Add metadata for stream @stream_name to sample
        """

        self.meta_info[stream_name] = meta_info

    def update_sample_meta_info(self, params: dict) -> None:
        """
        Update stream-independent meta information of sample (existing keys are overwritten)
        """
        self.sample_meta_info.update(params)

    def get_stream_data(self, stream_name: str) -> StreamData:
        """
        Get data for stream @stream_name from sample
        """
        assert self.streams_data.get(stream_name, -1) != -1, "stream name does not exist"
        return self.streams_data[stream_name]


class BatchSamples:
    """
    Container for source or target samples
    """

    samples: list[Sample]
    tokens_lens: torch.Tensor | None
    output_steps: int
    output_idxs: list[int]
    device: str | None

    def __init__(
        self, stream_names: list[str], num_samples: int, output_steps, output_idxs
    ) -> None:
        self.samples = [Sample(stream_names) for _ in range(num_samples)]
        self.tokens_lens = None
        self.output_steps = output_steps
        self.output_idxs = output_idxs
        self.device = None
        self.latent = []

    @property
    def batch_samples(self) -> typing.Self:
        return self

    def __len__(self) -> int:
        return len(self.samples)

    def to_device(
        self,
        device,
        target_steps: list[int] | None = None,
        include_source: bool = True,
        include_target_coords: bool = True,
        include_target_tokens: bool = True,
    ):
        for sample in self.samples:
            sample.to_device(
                device,
                target_steps,
                include_source,
                include_target_coords,
                include_target_tokens,
            )

        if include_source:
            self.tokens_lens = (
                self.tokens_lens.to(device, non_blocking=True)
                if self.tokens_lens is not None
                else None
            )

        self.device = device

        return self

    def device_slots(self) -> list[tuple[list | dict, int | str]]:
        """
        Where the tensors that to_device() moves by default are stored, see Sample.device_slots()
        """
        slots = [slot for sample in self.samples for slot in sample.device_slots()]
        slots.append((self.__dict__, "tokens_lens"))
        return slots

    def clear_target_coordinates(self, target_steps: list[int]) -> None:
        for sample in self.samples:
            sample.clear_target_coordinates(target_steps)

    def get_samples(self) -> list[Sample]:
        return self.samples

    def stack_sample_meta_info(self) -> dict[str, np.typing.NDArray]:
        """
        Stack the stream-independent meta information (e.g. diffusion noise level, timestamp)
        of all samples along the sample dim.
        Returns one array of length num_samples per key; empty if there are no samples.
        """
        if len(self.samples) == 0:
            return {}

        keys = set(self.samples[0].sample_meta_info)
        assert all(set(s.sample_meta_info) == keys for s in self.samples), (
            "all samples have to provide the same sample meta info keys, got "
            f"{[sorted(s.sample_meta_info) for s in self.samples]}"
        )

        stacked = {}
        for key in sorted(keys):
            values = [s.sample_meta_info[key] for s in self.samples]
            assert all(np.ndim(v) == 0 for v in values), (
                f"sample meta info '{key}' has to be a scalar per sample, "
                f"got shapes {[np.shape(v) for v in values]}"
            )
            stacked[key] = np.array(values)
        return stacked

    def get_subset(self, subset: list | None = None):
        if subset is None:
            return self
        else:
            assert len(list(set(subset))) == len(subset), "subset contains duplicates"
            # create copy and then select subset for samples and tokens_lens
            bs = copy.deepcopy(self)
            bs.samples = [bs.samples[i] for i in subset]
            torch_idxs = torch.tensor(subset, dtype=torch.long, device=bs.tokens_lens.device)
            bs.tokens_lens = torch.index_select(bs.tokens_lens, 1, torch_idxs)
            return bs

    def get_num_steps(self) -> int:
        """
        Get number of input/source steps from smallest of all available streams
        """
        # TODO: define explicitly
        lens = [
            len(stream.source_tokens_cells) for _, stream in self.samples[0].streams_data.items()
        ]

        return min(lens)

    def get_output_idxs(self) -> int:
        """
        Get forecast indices
        """
        return self.output_idxs

    def get_output_len(self) -> int:
        """
        Get length of output
        """
        return self.output_steps

    def get_device(self) -> str | torch.device:
        """
        Get device of tensors in the batch
        """
        return self.device

    def sources_empty(self) -> bool:
        """
        Check if sources for all samples are empty
        """
        return np.array([s.sources_empty() if s is not None else True for s in self.samples]).all()

    def targets_empty(self) -> bool:
        """
        Check if targets for all samples are empty
        """
        return np.array([s.targets_empty() if s is not None else True for s in self.samples]).all()

    def sources_nan(self) -> bool:
        """
        Check if sources for all samples are all NaN
        """
        return np.array([s.sources_nan() if s is not None else False for s in self.samples]).all()

    def targets_nan(self) -> bool:
        """
        Check if targets for all samples are all NaN
        """
        return np.array([s.targets_nan() if s is not None else False for s in self.samples]).all()

    def pin_memory(self):
        """Pin all tensors in this batch to CPU pinned memory"""

        # pin all samples
        for sample in self.samples:
            sample.pin_memory()

        # pin source_tokens_lens
        if isinstance(self.tokens_lens, torch.Tensor):
            self.tokens_lens = self.tokens_lens.pin_memory()

        return self


class ModelBatch:
    """
    Container for all data and metadata for one training batch.
    """

    # source samples (for model)
    source_samples: BatchSamples

    # target samples (for TargetAuxCalculator)
    target_samples: BatchSamples

    # index of corresponding target (for source samples) or source (for target samples)
    # these are in 1-to-1 corresponding for classical training modes (e.g. MTM, forecasting) but
    # can be more complex for strategies like student-teacher training
    source2target_matching_idxs: np.typing.NDArray[np.int32]
    target2source_matching_idxs: np.typing.NDArray[np.int32]

    # indices of valid outputs
    output_idxs: list[int]

    # device of the tensors in the batch
    device: str | torch.device

    # buffer that holds all tensors of the batch after pack(), None if not packed
    packed_buffer: torch.Tensor | None

    def __init__(
        self,
        stream_names: list[str],
        num_source_samples: int,
        num_target_samples: int,
        output_offset,
        output_steps,
    ) -> None:
        """ """

        # define forecast indices
        self.output_offset = output_offset
        self.output_steps = output_steps
        self.output_idxs = list(range(output_offset, output_steps))

        self.source_samples = BatchSamples(
            stream_names, num_source_samples, output_steps, self.output_idxs
        )
        self.target_samples = BatchSamples(
            stream_names, num_target_samples, output_steps, self.output_idxs
        )

        self.source2target_matching_idxs = np.full(num_source_samples, -1, dtype=np.int32)
        self.target2source_matching_idxs = [[] for _ in range(num_target_samples)]

        self.packed_buffer = None

    def set_source_target_matching(self, source_to_target: np.typing.NDArray) -> None:
        """
        Set the correspondence between source and target samples (identical for all streams)

        source_to_target: index of the corresponding target sample for each source sample
        """
        source_to_target = np.asarray(source_to_target, dtype=np.int32)
        assert len(source_to_target) == len(self.source_samples), (
            "source_to_target has to provide a target index for each source sample"
        )
        assert ((source_to_target >= 0) & (source_to_target < len(self.target_samples))).all(), (
            "invalid target sample index in source_to_target"
        )
        self.source2target_matching_idxs = source_to_target.copy()
        self.target2source_matching_idxs = [
            [sidx for sidx, t in enumerate(source_to_target) if t == tidx]
            for tidx in range(len(self.target_samples))
        ]

    def update_sample_meta_info(self, source_params: list[dict], target_params: list[dict]) -> None:
        """
        Update stream-independent meta information of all source and target samples
        """
        for sample, params in zip(self.source_samples.samples, source_params, strict=True):
            sample.update_sample_meta_info(params)
        for sample, params in zip(self.target_samples.samples, target_params, strict=True):
            sample.update_sample_meta_info(params)

    def pin_memory(self):
        """Pin all tensors in this batch to CPU pinned memory"""

        if self.packed_buffer is not None:
            self.packed_buffer = self._move_packed(lambda t: t.pin_memory())
            return self

        # pin source samples
        self.source_samples.pin_memory()

        # pin target samples
        self.target_samples.pin_memory()

        return self

    def pack(self):  # -> ModelBatch
        """
        Copy all tensors that to_device() moves into one buffer and replace them by views of it.

        A packed batch is transferred to the device with one host-to-device copy instead of one
        per tensor. Call it in the DataLoader worker: the buffer is allocated in shared memory,
        so sending the batch to the main process does not copy it again, and the main process
        maps and unmaps one segment per batch.

        The tensors on the device are then views of one allocation, which is only freed when
        the last of them is released.
        """

        slots = self.source_samples.device_slots() + self.target_samples.device_slots()

        # layout: offset of each distinct tensor in the buffer
        tensors = {}
        num_bytes = 0
        for container, key in slots:
            tensor = container[key]
            if _is_packable(tensor) and id(tensor) not in tensors:
                tensors[id(tensor)] = (tensor, num_bytes)
                size = tensor.numel() * tensor.element_size()
                num_bytes += -(-size // _PACK_ALIGNMENT) * _PACK_ALIGNMENT

        if num_bytes == 0:
            return self

        # torch has no public function to allocate in shared memory without copying;
        # default_collate uses the same call for its output buffer
        storage = torch.UntypedStorage._new_shared(num_bytes)  # pylint: disable=protected-access
        buffer = torch.empty(0, dtype=torch.uint8).set_(storage)

        views = {}
        for tensor_id, (tensor, offset) in tensors.items():
            size = tensor.numel() * tensor.element_size()
            view = buffer[offset : offset + size].view(tensor.dtype).view(tensor.shape)
            view.copy_(tensor)
            views[tensor_id] = view

        for container, key in slots:
            view = views.get(id(container[key]))
            if view is not None:
                container[key] = view

        self.packed_buffer = buffer

        return self

    def _move_packed(self, move: typing.Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Apply move (e.g. the transfer to a device) once to the packed buffer and replace the
        tensors of the batch by views of the result. Tensors that are not views of the packed
        buffer are moved one by one. Returns the moved buffer.
        """

        buffer = self.packed_buffer
        buffer_ptr = buffer.untyped_storage().data_ptr()
        moved = move(buffer)

        slots = self.source_samples.device_slots() + self.target_samples.device_slots()
        for container, key in slots:
            tensor = container[key]
            if not isinstance(tensor, torch.Tensor):
                continue
            if (
                _is_packable(tensor)
                and tensor.is_contiguous()
                and tensor.untyped_storage().data_ptr() == buffer_ptr
            ):
                offset = tensor.storage_offset() * tensor.element_size()
                size = tensor.numel() * tensor.element_size()
                container[key] = moved[offset : offset + size].view(tensor.dtype).view(tensor.shape)
            else:
                container[key] = move(tensor)

        return moved

    def to_device(self, device):  # -> ModelBatch
        """
        Move batch to device
        """

        if self.packed_buffer is not None:
            # one copy for the whole batch, see pack()
            self._move_packed(lambda t: t.to(device, non_blocking=True))
            self.packed_buffer = None
            self.source_samples.device = device
            self.target_samples.device = device
        else:
            self.source_samples.to_device(device)
            self.target_samples.to_device(device)

        self.device = device

        return self

    def to_device_for_chunked_inference(self, device, loss_steps: list[int]):
        """Move source inputs and the targets required for the final chunk loss."""
        self.source_samples.to_device(
            device,
            target_steps=[],
            include_target_coords=False,
            include_target_tokens=False,
        )
        self.target_samples.to_device(
            device,
            target_steps=loss_steps,
            include_source=False,
            include_target_coords=False,
        )
        self.device = device

        return self

    def to_device_for_output_chunk(self, device, target_steps: list[int]):
        """Move decoder coordinates for the active forecast chunk."""
        self.source_samples.to_device(
            device,
            target_steps=target_steps,
            include_source=False,
            include_target_tokens=False,
        )

        return self

    def clear_output_chunk_coordinates(self, target_steps: list[int]) -> None:
        """Release decoder coordinates after output for a forecast chunk was written."""
        self.source_samples.clear_target_coordinates(target_steps)

    def add_source_stream(
        self,
        source_sample_idx: int,
        stream_name: str,
        stream_data: StreamData,
        source_meta_info: SampleMetaData,
    ) -> None:
        """
        Add data for one stream to sample @source_sample_idx
        """

        self.source_samples.samples[source_sample_idx].add_stream_data(stream_name, stream_data)

        # add the meta_info
        self.source_samples.samples[source_sample_idx].add_meta_info(stream_name, source_meta_info)

    def add_target_stream(
        self,
        target_sample_idx: int,
        stream_name: str,
        stream_data: StreamData,
        target_meta_info: SampleMetaData,
    ) -> None:
        """
        Add data for one stream to sample @target_sample_idx
        """
        self.target_samples.samples[target_sample_idx].add_stream_data(stream_name, stream_data)

        # add the meta_info -- for target we have different
        self.target_samples.samples[target_sample_idx].add_meta_info(stream_name, target_meta_info)

    def is_empty(self):
        """
        Check if batch is empty
        """
        return self.source_samples.sources_empty() or self.target_samples.targets_empty()

    def is_nan(self):
        """
        Check if batch is all NaN
        """
        return self.source_samples.sources_nan() or self.target_samples.targets_nan()

    def sources_empty(self):
        """
        Check if batch sources are empty
        """
        return self.source_samples.sources_empty()

    def sources_nan(self):
        """
        Check if batch sources are all NaN
        """
        return self.source_samples.sources_nan()

    def targets_empty(self):
        """
        Check if batch targets are empty
        """
        return self.target_samples.targets_empty()

    def targets_nan(self):
        """
        Check if batch targets are all NaN
        """
        return self.target_samples.targets_nan()

    def len_sources(self) -> int:
        """
        Number of source samples
        """
        return len(self.source_samples)

    def len_targets(self) -> int:
        """
        Number of target samples
        """
        return len(self.target_samples)

    def get_source_sample(self, idx: int) -> Sample:
        """
        Get a source sample
        """
        return self.source_samples.samples[idx]

    def get_source_samples(self, subset: list | None = None) -> BatchSamples:
        """
        Get source samples
        """
        return self.source_samples.get_subset(subset)

    def get_target_sample(self, idx: int) -> Sample:
        """
        Get a target sample
        """
        return self.target_samples.samples[idx]

    def get_target_samples(self, subset: list | None = None) -> BatchSamples:
        """
        Get target samples
        """
        return self.target_samples.get_subset(subset)

    def get_source_idx_for_target(self, target_idx: int) -> int:
        """
        Get index of source sample for a given target sample index
        """
        return int(self.target2source_matching_idxs[target_idx])

    def get_target_idx_for_source(self, source_idx: int) -> int:
        """
        Get index of target sample for a given source sample index
        """
        return int(self.source2target_matching_idxs[source_idx])

    def get_output_idxs(self) -> int:
        """
        Get valid output steps
        """
        return self.output_idxs

    def get_output_len(self) -> int:
        """
        Get length of output
        """
        return self.output_steps

    def get_device(self) -> str | torch.device:
        """
        Get device of tensors in the batch
        """
        return self.device

    def get_num_source_steps(self) -> int:
        """
        Get number of input/source steps from smallest of all available streams
        """
        # TODO: define explicitly
        lens = [
            len(stream.source_tokens_cells)
            for _, stream in self.target_samples.samples[0].streams_data.items()
        ]

        return min(lens)

    def get_num_target_steps(self) -> int:
        """
        Get number of input/source steps from smallest of all available streams
        """
        # TODO: define explicitly
        # TODO: ensure that num_input_steps is constant across batch with different strategies
        lens = [
            len(stream.target_tokens)
            for _, stream in self.target_samples.samples[0].streams_data.items()
        ]

        return min(lens)
