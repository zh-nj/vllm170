# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Approximate fast prefill: bounded decoder SWA replay on a query suffix.

Ported from FastLLM's ``FASTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY``
(``src/models/deepseekv41.cpp``): layers below ``tail_layer`` compute every
token of a prefill chunk, including the full compressed KV, indexer K, top-k
indices and candidate blocks that the later layers read. From ``tail_layer``
on, only the chunk's *query side* is bounded -- the last
``min(chunk_tokens, sliding_window)`` rows of each request, plus every decode
row.

The approximation is confined to the local sliding window of the bounded rows:
their window starts at the clip boundary instead of reaching into history the
tail layers never computed (``windowStart = 0`` at
``src/models/deepseekv41.cpp:4146``). RoPE, the indexer's candidate selection
and the compressed KV keep true absolute positions, so the bounded rows see the
same full-context compressed history as an unbounded run; only their local
attention and everything downstream of it (hidden, logits) changes.

This module owns the CPU-side row plan and its validation. The device-side
gather/scatter and the metadata swap live in the model and model state.
"""

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)

_SWITCH = "dsv41_fast_prefill"


def fast_prefill_enabled(config: VllmConfig) -> bool:
    """Whether the bounded decoder tail is requested, validating its scope.

    Raises:
        ValueError: The requested configuration is out of the supported scope.
    """
    extra = config.additional_config
    if not (isinstance(extra, dict) and bool(extra.get(_SWITCH, False))):
        return False
    if extra.get("ced_prefill", False):
        raise ValueError(
            "dsv41_fast_prefill and ced_prefill both rewrite prefill execution "
            "and cannot be enabled together"
        )
    if not config.use_v2_model_runner:
        raise ValueError("dsv41_fast_prefill requires the v2 model runner")
    parallel = config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.use_ubatching:
        raise ValueError("dsv41_fast_prefill currently requires TP1 without DBO")
    tail_layer(config.model_config.hf_config)
    return True


def tail_layer(config) -> int:
    """First layer whose queries may be bounded: one past the last KV source.

    FastLLM's ``decoderSwaTailLayer``. Everything the later layers read is
    produced by layers at or below the last KV source, so the clip can only
    start after it.

    Raises:
        ValueError: The layer span after the last KV source cannot be bounded.
    """
    sources = [int(layer) for layer in getattr(config, "kv_source_layer_ids", ())]
    if not sources:
        raise ValueError(
            "dsv41_fast_prefill needs kv_source_layer_ids to locate the bounded tail"
        )
    tail = max(sources) + 1
    ratios = getattr(config, "compress_ratios", None)
    if ratios is None:
        raise ValueError("dsv41_fast_prefill needs compress_ratios")
    for layer in range(tail, int(config.num_hidden_layers)):
        if int(ratios[layer]) > 1:
            raise ValueError(
                f"dsv41_fast_prefill cannot bound layer {layer} with "
                f"compress_ratio={ratios[layer]}; only ratio 0 and 1 tails are "
                "supported"
            )
    return tail


@dataclass
class TailRows:
    """The rows the bounded tail layers compute, and how they map back.

    ``keep`` is ascending, so a request's kept rows stay contiguous and in
    order; every request keeps at least its last row, which is the row the
    sampler reads.
    """

    keep: np.ndarray
    """[num_tokens] int64 original batch rows the tail layers compute."""

    query_start_loc: np.ndarray
    """[num_reqs + 1] int32 offsets into ``keep``, one per batch request."""

    gather_clamp: np.ndarray
    """[num_prefills] int32 upper bound for the prefill SWA gather, or -1 to
    keep the unbounded value (a chunk that was not truncated still reads its
    real history window)."""

    truncated: np.ndarray
    """[num_reqs] bool, True where this request's chunk was truncated."""

    def __post_init__(self) -> None:
        if self.keep.ndim != 1 or self.query_start_loc.ndim != 1:
            raise ValueError("tail row plan expects flat index arrays")
        if int(self.query_start_loc[0]) != 0:
            raise ValueError("tail row plan query_start_loc must start at zero")
        if int(self.query_start_loc[-1]) != self.keep.size:
            raise ValueError("tail row plan query_start_loc must end at num_tokens")
        if np.any(np.diff(self.keep) <= 0):
            raise ValueError("tail row plan rows must ascend without repeats")

    @property
    def num_tokens(self) -> int:
        return int(self.keep.size)

    @property
    def num_reqs(self) -> int:
        return int(self.query_start_loc.size - 1)

    @property
    def num_truncated(self) -> int:
        return int(self.truncated.sum())

    def to_device(self, device: torch.device) -> "TailRowsDevice":
        return TailRowsDevice(
            keep=torch.from_numpy(self.keep).to(device=device, dtype=torch.int64),
            query_start_loc=torch.from_numpy(self.query_start_loc).to(
                device=device, dtype=torch.int32
            ),
            gather_clamp=torch.from_numpy(self.gather_clamp).to(
                device=device, dtype=torch.int32
            ),
        )


@dataclass(frozen=True)
class TailRowsDevice:
    keep: torch.Tensor
    query_start_loc: torch.Tensor
    gather_clamp: torch.Tensor


def plan_tail_rows(
    query_start_loc_np: np.ndarray,
    is_prefilling_np: np.ndarray,
    window_size: int,
    *,
    num_decode_tokens: int,
) -> TailRows | None:
    """Choose the rows the tail layers compute, or None to compute everything.

    Args:
        query_start_loc_np: [num_reqs + 1] batch row offsets, decodes first.
        is_prefilling_np: [num_reqs] still-prefilling flags.
        window_size: Local sliding window width.
        num_decode_tokens: Rows belonging to non-prefilling requests.

    Returns:
        The row plan, or None when no request's chunk exceeds the window (the
        case FastLLM leaves entirely alone).
    """
    query_start_loc_np = np.asarray(query_start_loc_np, dtype=np.int64)
    is_prefilling_np = np.asarray(is_prefilling_np, dtype=bool)
    num_reqs = int(is_prefilling_np.size)
    if query_start_loc_np.size != num_reqs + 1:
        raise ValueError("query_start_loc must have num_reqs + 1 entries")
    if window_size <= 0:
        raise ValueError("dsv41_fast_prefill needs a positive sliding window")

    # The metadata builders split the batch as [decodes][prefills] and index the
    # prefill SWA gather by prefill position, so the row layout has to match.
    decode_rows = int(
        sum(
            int(query_start_loc_np[i + 1]) - int(query_start_loc_np[i])
            for i in range(num_reqs)
            if not is_prefilling_np[i]
        )
    )
    first_prefill = next((i for i in range(num_reqs) if is_prefilling_np[i]), num_reqs)
    if decode_rows != num_decode_tokens or not is_prefilling_np[first_prefill:].all():
        raise ValueError(
            "dsv41_fast_prefill expects the batch layout [decodes][prefills]"
        )

    keep_parts: list[np.ndarray] = []
    starts = [0]
    gather_clamp: list[int] = []
    truncated = np.zeros(num_reqs, dtype=bool)
    for index in range(num_reqs):
        begin = int(query_start_loc_np[index])
        end = int(query_start_loc_np[index + 1])
        if end <= begin:
            raise ValueError(f"request {index} has no scheduled rows")
        if not is_prefilling_np[index]:
            keep_parts.append(np.arange(begin, end, dtype=np.int64))
            starts.append(starts[-1] + (end - begin))
            continue
        count = end - begin
        kept = min(count, window_size)
        keep_parts.append(np.arange(end - kept, end, dtype=np.int64))
        gather_clamp.append(kept if kept < count else -1)
        truncated[index] = kept < count
        starts.append(starts[-1] + kept)

    if not truncated.any():
        return None

    keep = np.concatenate(keep_parts)
    return TailRows(
        keep=keep,
        query_start_loc=np.asarray(starts, dtype=np.int32),
        gather_clamp=np.asarray(gather_clamp, dtype=np.int32),
        truncated=truncated,
    )


def plan_debug(plan: TailRows, query_start_loc_np: np.ndarray) -> str:
    """One-line summary of a plan, for the enable log and diagnostics."""
    sizes = []
    for index in range(plan.num_reqs):
        begin = int(query_start_loc_np[index])
        end = int(query_start_loc_np[index + 1])
        kept = int(plan.query_start_loc[index + 1] - plan.query_start_loc[index])
        sizes.append(f"{end - begin}->{kept}")
    return (
        f"rows {int(query_start_loc_np[-1])}->{plan.num_tokens} over "
        f"{plan.num_reqs} requests ({', '.join(sizes)}), "
        f"{plan.num_truncated} truncated"
    )


@dataclass
class TailBufferSwap:
    """Module attributes replaced for the bounded span, and how to put them back.

    Restoring matters: the next step may be a decode step whose rows are read
    from the unbounded buffers.
    """

    saved: list[tuple[Any, str, torch.Tensor]]

    def restore(self) -> None:
        for module, attr, value in self.saved:
            setattr(module, attr, value)


@dataclass
class FastPrefillStep:
    """One step's bounded tail: which rows run, and the metadata they run with.

    ``metadata`` holds entries only for layer prefixes at or after
    ``tail_layer``; the model merges it over the unbounded metadata dict when it
    reaches the boundary, so layers below it keep the batch they were built for.
    """

    tail_layer: int
    plan: TailRowsDevice
    metadata: dict[str, Any]
    positions: torch.Tensor
    input_ids: torch.Tensor
    num_tokens: int

    def bind_tail_buffers(self, model: Any) -> "TailBufferSwap | None":
        """Point the bounded layers at the gathered top-k and candidate rows.

        Both buffers are indexed by prefill-region row, and the bounded batch
        renumbers rows, so without this the tail reads the top-k and candidate
        entries of rows it did not keep -- entries that describe a different
        query. FastLLM gathers both at the same offset as hidden and HC pre-mix
        (``trimRows(segTopK)`` / ``trimRows(segCandidate)``,
        ``deepseekv41.cpp:3790-3800``). Only attributes that are literally the
        model's buffers are swapped, so nothing else is disturbed.
        """
        topk = getattr(model, "topk_indices_buffer", None)
        if topk is None:
            return None
        keep = self.plan.keep
        replacements = [
            ("topk_indices_buffer", topk, topk.index_select(0, keep).contiguous())
        ]
        candidates = getattr(model, "candidate_block_buffer", None)
        if candidates is not None:
            gathered = candidates.index_select(0, keep).contiguous()
            replacements += [
                ("candidate_blocks", candidates, gathered),
                ("candidate_block_buffer", candidates, gathered),
            ]
        saved: list[tuple[Any, str, torch.Tensor]] = []
        for index, layer in enumerate(model.layers):
            if index < self.tail_layer:
                continue
            attn = getattr(layer, "attn", None)
            if attn is None:
                continue
            indexer = getattr(attn, "indexer", None)
            modules = (attn, indexer, getattr(indexer, "indexer_op", None))
            for module in modules:
                if module is None:
                    continue
                for attr, source, replacement in replacements:
                    if getattr(module, attr, None) is source:
                        saved.append((module, attr, source))
                        setattr(module, attr, replacement)
        return TailBufferSwap(saved)

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Select the bounded rows of a full-width batch tensor."""
        return tensor.index_select(0, self.plan.keep)

    def scatter(self, tensor: torch.Tensor, width: int) -> torch.Tensor:
        """Place bounded rows back at their batch positions, zero elsewhere.

        The zeroed rows are never read: consumers either index the bounded rows
        (the sampler's logits indices, the drafter's context) or are themselves
        bounded on this rank. They exist so the PP message and the runner's
        output keep the shapes they had without the switch.
        """
        out = torch.zeros(
            (width, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
        )
        return out.index_copy_(0, self.plan.keep, tensor)


class FastPrefillMetadata:
    """Builds the bounded attention metadata for the layers at or after the tail.

    Mirrors ``CEDMetadata``: the attention groups are cloned with their layer
    list filtered to the bounded span, because ``build_attn_groups`` files
    KV-sharing layers into their source's group, so a single group spans both
    sides of the boundary.
    """

    def __init__(self, config: VllmConfig, device: torch.device, tail: int):
        self.config = config
        self.device = device
        self.tail_layer = tail
        self.window = int(config.model_config.hf_config.sliding_window)
        logger.info_once(
            "dsv41_fast_prefill: approximate bounded decoder SWA replay, "
            "boundary layer %d, window %d",
            tail,
            self.window,
        )

    def prepare(
        self,
        input_batch: Any,
        block_tables: list[torch.Tensor],
        slot_mappings: torch.Tensor,
        groups: list[list[AttentionGroup]],
        kv_cache_config: Any,
    ) -> FastPrefillStep | None:
        """Return this step's bounded tail, or None to compute the full batch."""
        if not input_batch.has_prefill:
            return None
        if not self._has_bounded_layers(groups):
            # A rank that only holds layers below the boundary has nothing to
            # bound: its chunk flows through unchanged.
            return None
        is_prefilling = np.asarray(input_batch.is_prefilling_np, dtype=bool)
        # The metadata builders split the batch as [decodes][prefills]; the row
        # counts of the non-prefilling requests are exactly that decode span.
        starts = np.asarray(input_batch.query_start_loc_np, dtype=np.int64)
        num_decode_tokens = int(
            sum(
                int(starts[i + 1]) - int(starts[i])
                for i in range(is_prefilling.size)
                if not is_prefilling[i]
            )
        )
        plan = plan_tail_rows(
            starts,
            is_prefilling,
            self.window,
            num_decode_tokens=num_decode_tokens,
        )
        if plan is None:
            return None

        device_plan = plan.to_device(self.device)
        keep = device_plan.keep
        positions = input_batch.positions.index_select(0, keep)
        input_ids = input_batch.input_ids.index_select(0, keep)
        metadata = self._build(
            plan,
            device_plan,
            input_batch,
            block_tables,
            slot_mappings,
            kv_cache_config,
            groups,
        )
        self._clamp_gather_lens(metadata, device_plan)
        return FastPrefillStep(
            tail_layer=self.tail_layer,
            plan=device_plan,
            metadata=metadata,
            positions=positions,
            input_ids=input_ids,
            num_tokens=plan.num_tokens,
        )

    def _clamp_gather_lens(
        self, metadata: dict[str, Any], device_plan: TailRowsDevice
    ) -> None:
        """Stop a truncated request's local gather before the clip boundary.

        FastLLM drops the window KV entirely for a clipped segment
        (``windowStart = 0``); here the same bound falls out of capping the
        gather at the rows this layer actually wrote. Requests that were not
        truncated keep the unbounded length and still read their real history.
        """
        clamp = device_plan.gather_clamp
        bounded = clamp >= 0
        if not bool(bounded.any()):
            return
        sentinel = torch.iinfo(torch.int32).max
        limit = torch.where(bounded, clamp, sentinel)
        for value in metadata.values():
            lengths = getattr(value, "prefill_gather_lens", None)
            if lengths is None:
                continue
            value.prefill_gather_lens = torch.minimum(lengths, limit)

    def _build(
        self,
        plan: TailRows,
        device_plan: TailRowsDevice,
        input_batch: Any,
        block_tables: list[torch.Tensor],
        slot_mappings: torch.Tensor,
        kv_cache_config: Any,
        groups: list[list[AttentionGroup]],
    ) -> dict[str, Any]:
        query_start_loc_cpu = torch.from_numpy(plan.query_start_loc)
        seq_lens = input_batch.seq_lens[: plan.num_reqs]
        upper_bound = input_batch.seq_lens_cpu_upper_bound[: plan.num_reqs]
        # Every group is rebuilt, with its full layer list, because a bounded
        # layer reads caches whose metadata is keyed by a *source* layer below
        # the boundary: the compressed cache (``compressed_cache_prefix``) and
        # the indexer K cache, which a non-owning index source shares with its kv
        # source (``attention.py:1145``). Filtering the layer names dropped those
        # keys, so the tail read the unbounded entry -- metadata describing 2044
        # query rows against 128 rows of activations, which faulted the indexer.
        # Groups no bounded layer reads are rebuilt too; the extra cost is one
        # metadata build per group per step.
        return build_attn_metadata(
            attn_groups=groups,
            num_reqs=plan.num_reqs,
            num_tokens=plan.num_tokens,
            query_start_loc_gpu=device_plan.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=int(np.diff(plan.query_start_loc).max()),
            seq_lens=seq_lens,
            max_seq_len=int(upper_bound.max().item()),
            block_tables=block_tables,
            slot_mappings=slot_mappings[:, device_plan.keep].contiguous(),
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=upper_bound,
            positions=input_batch.positions.index_select(0, device_plan.keep),
            is_prefilling=torch.from_numpy(
                np.asarray(input_batch.is_prefilling_np, dtype=bool)
            ),
        )

    def _has_bounded_layers(self, groups: list[list[AttentionGroup]]) -> bool:
        """Whether any layer this rank holds sits at or after the boundary."""
        return any(
            extract_layer_index(name) >= self.tail_layer
            for collection in groups
            for group in collection
            for name in group.layer_names
        )


_ROW_LOG_TRIGGER = "/tmp/dsv41_fastprefill_rows_on"
_ROW_LOG_BASE = "/tmp/dsv41_fastprefill_rows"


def record_layer_rows(rank: int, layer: int, rows: int, tail: int) -> None:
    """Append one layer's executed row count while the trigger file exists.

    The port's claim is per layer: everything below the tail computes the whole
    chunk, everything from the tail on computes only the bounded suffix. Nothing
    in the engine reports that, and the worker processes do not inherit extra
    ``VLLM_*`` names, so the switch is a file -- the same reason the row probe
    uses one. The step id comes from a second file the driver bumps before each
    measured request, which keeps the rows of one request groupable.

    Args:
        rank: Pipeline rank that ran the layer.
        layer: Global layer index.
        rows: Rows this layer actually processed.
        tail: First layer allowed to be bounded.
    """
    if not os.path.exists(_ROW_LOG_TRIGGER):
        return
    try:
        with open(f"{_ROW_LOG_BASE}.step") as handle:
            step = int(handle.read().strip())
    except (OSError, ValueError):
        return
    with open(f"{_ROW_LOG_BASE}.rank{rank}.tsv", "a") as handle:
        handle.write(f"{step}\t{layer}\t{rows}\t{int(layer >= tail)}\n")
