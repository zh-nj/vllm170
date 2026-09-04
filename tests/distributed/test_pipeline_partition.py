# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

from vllm.distributed.utils import get_pp_indices


def test_custom_layer_partition(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:

        def _verify(partition_str, num_layers, pp_size, goldens):
            bak = os.environ.get("VLLM_PP_LAYER_PARTITION", None)
            m.setenv("VLLM_PP_LAYER_PARTITION", partition_str)
            for pp_rank, golden in enumerate(goldens):
                assert get_pp_indices(num_layers, pp_rank, pp_size) == golden
            if bak is not None:
                m.setenv("VLLM_PP_LAYER_PARTITION", bak)

        # Even partition
        _verify("5,5,5,5", 20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        # Balanced partition
        _verify("4,6,6,4", 20, 4, [(0, 4), (4, 10), (10, 16), (16, 20)])
        # Put reminder somewhere
        _verify("5,6,5,6", 22, 4, [(0, 5), (5, 11), (11, 16), (16, 22)])
        # Invalid partition strings
        with pytest.raises(ValueError):
            _verify("5,5,5,5,", 20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        with pytest.raises(ValueError):
            _verify("5,5,5,a", 20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        # Wrong number of partitions
        with pytest.raises(ValueError):
            _verify("5,5,5", 20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        # Wrong number of layers
        with pytest.raises(ValueError):
            _verify("5,5,5,5", 21, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])


def test_default_cmp170hx_layer_partition(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    assert [get_pp_indices(43, rank, 3) for rank in range(3)] == [
        (0, 15),
        (15, 31),
        (31, 43),
    ]


def test_explicit_layer_partition_overrides_cmp170hx_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "14,15,14")
    assert [get_pp_indices(43, rank, 3) for rank in range(3)] == [
        (0, 14),
        (14, 29),
        (29, 43),
    ]


@pytest.mark.parametrize(
    "num_hidden_layers,pp_size,pp_rank,indices",
    [
        # pp_size 2
        (2, 2, 0, (0, 1)),
        (2, 2, 1, (1, 2)),
        (3, 2, 0, (0, 2)),
        (3, 2, 1, (2, 3)),
        # pp_size 3
        (3, 3, 0, (0, 1)),
        (3, 3, 1, (1, 2)),
        (3, 3, 2, (2, 3)),
        (4, 3, 0, (0, 1)),
        (4, 3, 1, (1, 3)),
        (4, 3, 2, (3, 4)),
        (5, 3, 0, (0, 2)),
        (5, 3, 1, (2, 4)),
        (5, 3, 2, (4, 5)),
    ],
)
def test_uneven_auto_partition(
    num_hidden_layers: int,
    pp_size: int,
    pp_rank: int,
    indices: tuple[int, int],
):
    assert indices == get_pp_indices(num_hidden_layers, pp_rank, pp_size)
