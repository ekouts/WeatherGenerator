# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from weathergen.datasets.batch import ModelBatch, SampleMetaData


def test_model_batch_reports_temporal_index_and_unique_tensor_storage_bytes() -> None:
    batch = ModelBatch(["stream"], 1, 1, 0, 1, temporal_index=17)
    shared = torch.arange(8, dtype=torch.float32)
    separate = torch.ones(3, dtype=torch.int32)

    batch.source_samples.tokens_lens = shared
    batch.target_samples.tokens_lens = shared[2:]
    batch.source_samples.samples[0].meta_info["stream"] = SampleMetaData(params={}, mask=separate)

    assert batch.temporal_index == 17
    assert batch.unique_tensor_storage_bytes() == (
        shared.untyped_storage().nbytes() + separate.untyped_storage().nbytes()
    )
