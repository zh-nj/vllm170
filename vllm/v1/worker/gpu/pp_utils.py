# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


class PPRecvBufferState(Enum):
    FREE = "free"
    RECEIVING = "receiving"
    MODEL = "model"
    DSPARK = "dspark"


class PPRecvBufferGuard:
    """Tracks enqueue-time ownership of a reused PP receive buffer."""

    def __init__(self) -> None:
        self._state = PPRecvBufferState.FREE
        self._generation = 0
        self._owner_stream: object | None = None

    @property
    def state(self) -> PPRecvBufferState:
        return self._state

    @property
    def is_stream_tracked(self) -> bool:
        return self._owner_stream is not None

    def begin_receive(self, stream: object | None = None) -> int:
        if self._owner_stream is not None:
            raise RuntimeError(
                "Cannot start a receive while the PP receive buffer still has "
                f"owner stream {self._owner_stream}."
            )
        self._transition(
            PPRecvBufferState.FREE,
            PPRecvBufferState.RECEIVING,
            "start a receive",
        )
        self._generation += 1
        self._owner_stream = stream
        return self._generation

    def finish_receive(self, generation: int, stream: object | None = None) -> None:
        if generation != self._generation:
            raise RuntimeError(
                "Stale PP receive completion for buffer generation "
                f"{generation}; current generation is {self._generation}."
            )
        self._assert_stream(stream, "finish a receive")
        self._transition(
            PPRecvBufferState.RECEIVING,
            PPRecvBufferState.MODEL,
            "finish a receive",
        )

    def begin_dspark(self, stream: object | None = None) -> None:
        self._assert_stream(stream, "start DSpark")
        self._transition(
            PPRecvBufferState.MODEL,
            PPRecvBufferState.DSPARK,
            "start DSpark",
        )

    def release(self, stream: object | None = None) -> None:
        self._assert_stream(stream, "release the buffer")
        if self._state not in (
            PPRecvBufferState.MODEL,
            PPRecvBufferState.DSPARK,
        ):
            raise RuntimeError(
                f"Cannot release PP receive buffer while it is {self._state.value}."
            )
        self._state = PPRecvBufferState.FREE
        self._owner_stream = None

    def _assert_stream(self, stream: object | None, action: str) -> None:
        if self._owner_stream is not None and stream != self._owner_stream:
            raise RuntimeError(
                f"Cannot {action} on CUDA stream {stream}; PP receive buffer "
                f"is owned by stream {self._owner_stream}."
            )

    def _transition(
        self,
        expected: PPRecvBufferState,
        new: PPRecvBufferState,
        action: str,
    ) -> None:
        if self._state is not expected:
            raise RuntimeError(
                f"Cannot {action} while PP receive buffer is {self._state.value}; "
                f"expected {expected.value}."
            )
        self._state = new


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    # Proposed tokens from the last rank for speculative decoding over PP.
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    that produce a sampled token this step, and therefore must have that token
    (and the draft block proposed from it) propagated to the earlier PP stages.
    Returns None if no request in the batch produces a sample."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None


def pad_sampled_tokens(
    sampled_token_ids: torch.Tensor, max_sample_len: int
) -> torch.Tensor:
    """Pad a sampled-token matrix to the fixed PP collective width."""
    width = sampled_token_ids.shape[-1]
    if width == max_sample_len:
        return sampled_token_ids
    if width > max_sample_len:
        raise ValueError(
            f"sample width {width} exceeds PP buffer width {max_sample_len}"
        )
    padded = sampled_token_ids.new_full(
        (sampled_token_ids.shape[0], max_sample_len), -1
    )
    padded[:, :width] = sampled_token_ids
    return padded


def _snapshot_draft_tokens_for_pp(draft_tokens: torch.Tensor) -> torch.Tensor:
    return draft_tokens.to(
        dtype=torch.int64,
        memory_format=torch.contiguous_format,
        copy=True,
    )


def pack_speculative_pp_relay(
    sampled_token_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    max_sample_len: int,
) -> torch.Tensor:
    """Pack speculative PP results into one fixed-width collective."""
    sampled_token_ids = pad_sampled_tokens(sampled_token_ids, max_sample_len)
    expected_draft_width = max_sample_len - 1
    if draft_tokens.ndim != 2 or draft_tokens.shape != (
        sampled_token_ids.shape[0],
        expected_draft_width,
    ):
        raise ValueError(
            "PP draft-token shape must be "
            f"[{sampled_token_ids.shape[0]}, {expected_draft_width}], "
            f"got {tuple(draft_tokens.shape)}"
        )
    counts = torch.stack((num_sampled, num_rejected), dim=1).to(torch.int64)
    return torch.cat((sampled_token_ids, draft_tokens.to(torch.int64), counts), dim=1)


def unpack_speculative_pp_relay(
    relay: torch.Tensor, max_sample_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sampled tokens, counts, and proposals from a packed PP relay."""
    draft_end = max_sample_len * 2 - 1
    sampled_tokens = relay[:, :max_sample_len]
    draft_tokens = relay[:, max_sample_len:draft_end]
    num_sampled = relay[:, draft_end].to(torch.int32).contiguous()
    num_rejected = relay[:, draft_end + 1].to(torch.int32).contiguous()
    return sampled_tokens, num_sampled, num_rejected, draft_tokens


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
        relay_draft_tokens: bool = False,
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.relay_draft_tokens = relay_draft_tokens
        if relay_draft_tokens and num_speculative_steps < 1:
            raise ValueError("Draft-token relay requires speculative decoding.")
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)
        self._pending_speculative_relay: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
            draft_tokens=slot.draft_tokens,
        )

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            if self.relay_draft_tokens:
                relay = torch.empty(
                    num_reqs,
                    self.max_sample_len * 2 + 1,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    relay, src=self.last_rank, group=self.broadcast_group
                )
                sampled_tokens, num_sampled, num_rejected, draft_tokens = (
                    unpack_speculative_pp_relay(relay, self.max_sample_len)
                )
                relay.record_stream(self.main_stream)
                num_sampled.record_stream(self.main_stream)
                num_rejected.record_stream(self.main_stream)
            else:
                sampled_tokens = torch.empty(
                    num_reqs,
                    self.max_sample_len,
                    dtype=torch.int64,
                    device=self.device,
                )
                combined = torch.empty(
                    2, num_reqs, dtype=torch.int32, device=self.device
                )
                torch.distributed.broadcast(
                    sampled_tokens, src=self.last_rank, group=self.broadcast_group
                )
                torch.distributed.broadcast(
                    combined, src=self.last_rank, group=self.broadcast_group
                )
                num_sampled, num_rejected = combined.unbind(dim=0)
                draft_tokens = None
                sampled_tokens.record_stream(self.main_stream)
                combined.record_stream(self.main_stream)
            event = self.broadcast_stream.record_event()
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        # The receiver always posts a fixed-width buffer. Prefill and the first
        # decode step may produce only one sampled token, so pad before issuing
        # the collective to keep NCCL element counts identical on every rank.
        sampled_token_ids = pad_sampled_tokens(sampled_token_ids, self.max_sample_len)

        if self.relay_draft_tokens:
            if self._pending_speculative_relay is not None:
                raise RuntimeError("Previous speculative PP relay is still pending.")
            self._pending_speculative_relay = (
                sampled_token_ids,
                num_sampled,
                num_rejected,
            )
            return

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                sampled_token_ids.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)

    def broadcast_draft(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        """Relay speculative proposals from the last rank to target PP stages."""
        assert self.is_last_rank
        if not self.relay_draft_tokens:
            raise RuntimeError("Draft-token relay is not enabled for this PP handler.")
        if compute_need_sampled_mask(input_batch) is None:
            return
        pending = self._pending_speculative_relay
        if pending is None:
            raise RuntimeError("Speculative PP relay has no sampled outputs.")
        self._pending_speculative_relay = None
        sampled_token_ids, num_sampled, num_rejected = pending
        relay = pack_speculative_pp_relay(
            sampled_token_ids,
            draft_tokens,
            num_sampled,
            num_rejected,
            self.max_sample_len,
        )
        if current_platform.is_xpu():
            self.main_stream.synchronize()
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                relay, src=self.last_rank, group=self.broadcast_group
            )
            relay.record_stream(self.broadcast_stream)
