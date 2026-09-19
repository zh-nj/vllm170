# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    RoutedExpertsTensors,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors
from vllm.v1.worker.utils import raise_if_nan_logits

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


@dataclass(frozen=True)
class StepTimingSample:
    forward_ms: float
    drafter_ms: float
    num_target_tokens: int
    num_reqs: int
    full_cudagraph: bool


def _timing_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


@dataclass
class StepTimingEvents:
    prepare_start: torch.cuda.Event = field(default_factory=_timing_event)
    prepare_end: torch.cuda.Event = field(default_factory=_timing_event)
    attn_meta_start: torch.cuda.Event | None = None
    attn_meta_end: torch.cuda.Event | None = None
    inputs_start: torch.cuda.Event | None = None
    inputs_end: torch.cuda.Event | None = None
    forward_start: torch.cuda.Event = field(default_factory=_timing_event)
    forward_end: torch.cuda.Event = field(default_factory=_timing_event)
    drafter_start: torch.cuda.Event = field(default_factory=_timing_event)
    drafter_end: torch.cuda.Event = field(default_factory=_timing_event)


class StepTimingCollector:
    """Times the steps run inside ``collect``; record calls no-op outside it.

    Every step gets its own events, so the steps queue back-to-back and the
    block resolves them all behind one sync on the way out.
    """

    def __init__(self):
        self._collecting = False
        self._step: StepTimingEvents | None = None
        self._batch = (False, 0, 0)
        self._timed: list[tuple[StepTimingEvents, tuple[bool, int, int]]] = []
        # Streaming mode: resolve the events every N steps while serving and log
        # the split, instead of resolving one profiled block on the way out.
        self._stream_every = 0
        self._stream: list[tuple[float, float, float, float]] = []
        self._last_forward_start: torch.cuda.Event | None = None

    def set_stream_interval(self, every: int) -> None:
        """Log target/draft step timings every ``every`` steps. 0 disables."""
        self._stream_every = max(0, every)
        if self._stream_every:
            self._collecting = True

    def _resolve_stream(self) -> None:
        timed, self._timed = self._timed, []
        timed[-1][0].drafter_end.synchronize()

        def span(start: torch.cuda.Event | None, end: torch.cuda.Event | None) -> float:
            if start is None or end is None:
                return 0.0
            return start.elapsed_time(end)

        for events, _ in timed:
            # Step period: one step's forward_start to the next one's, which is
            # the only in-process measure of the whole engine period.
            if self._last_forward_start is None:
                period = 0.0
            else:
                period = self._last_forward_start.elapsed_time(events.forward_start)
            self._last_forward_start = events.forward_start
            self._stream.append(
                (
                    events.forward_start.elapsed_time(events.forward_end),
                    events.drafter_start.elapsed_time(events.drafter_end),
                    period,
                    events.prepare_start.elapsed_time(events.prepare_end),
                    span(events.attn_meta_start, events.attn_meta_end),
                    span(events.inputs_start, events.inputs_end),
                )
            )
        if len(self._stream) < self._stream_every:
            return
        samples, self._stream = self._stream, []
        forward = sorted(s[0] for s in samples)
        draft = sorted(s[1] for s in samples)
        period = sorted(s[2] for s in samples)
        prep = sorted(s[3] for s in samples)
        attn = sorted(s[4] for s in samples)
        inputs = sorted(s[5] for s in samples)
        mid = len(samples) // 2
        logger.info(
            "step timing over %d steps: target p50 %.2f ms (min %.2f), "
            "draft p50 %.2f ms (min %.2f), prep p50 %.2f ms "
            "(attn_meta %.2f, inputs %.2f), period p50 %.2f ms (other %.2f ms)",
            len(samples),
            forward[mid],
            forward[0],
            draft[mid],
            draft[0],
            prep[mid],
            attn[mid],
            inputs[mid],
            period[mid],
            period[mid] - forward[mid] - draft[mid] - prep[mid],
        )

    @contextlib.contextmanager
    def collect(self) -> Iterator[list[StepTimingSample]]:
        """Time every step run in this block.

        The yielded list holds one sample per timed step once the block exits;
        it stays empty inside the block, where the timings are still on device.
        """
        samples: list[StepTimingSample] = []
        self._collecting = True
        try:
            yield samples
        finally:
            self._collecting = False
            timed, self._timed, self._step = self._timed, [], None
        if not timed:
            return
        # Same stream, issue order: once the last step is done, so are the rest.
        timed[-1][0].drafter_end.synchronize()
        samples.extend(
            StepTimingSample(
                events.forward_start.elapsed_time(events.forward_end),
                events.drafter_start.elapsed_time(events.drafter_end),
                num_target_tokens,
                num_reqs,
                full_cudagraph,
            )
            for events, (full_cudagraph, num_target_tokens, num_reqs) in timed
        )

    def record_batch(self, input_batch: "InputBatch", full_cudagraph: bool) -> None:
        """Costs from different execution modes must not share a cost curve."""
        self._batch = (full_cudagraph, input_batch.num_tokens, input_batch.num_reqs)

    def forward_start(self) -> None:
        if self._collecting:
            self._step = StepTimingEvents()
            pending = getattr(self, "_pending", None) or {}
            self._pending = None
            prepare_start = pending.get("prepare_start")
            if prepare_start is not None:
                self._step.prepare_start = prepare_start
            else:
                # No prepare_start this step: keep the pair valid and zero.
                self._step.prepare_start.record()
            for name in ("attn_meta", "inputs"):
                start = pending.get(f"{name}_start")
                end = pending.get(f"{name}_end")
                if start is not None and end is not None:
                    setattr(self._step, f"{name}_start", start)
                    setattr(self._step, f"{name}_end", end)
            self._step.prepare_end.record()
            self._step.forward_start.record()

    def _mark(self, name: str) -> None:
        if self._collecting:
            if getattr(self, "_pending", None) is None:
                self._pending = {}
            event = _timing_event()
            event.record()
            self._pending[name] = event

    def prepare_start(self) -> None:
        """Opens the step before the target forward: metadata build and inputs."""
        self._mark("prepare_start")

    def attn_meta_start(self) -> None:
        self._mark("attn_meta_start")

    def attn_meta_end(self) -> None:
        self._mark("attn_meta_end")

    def inputs_start(self) -> None:
        self._mark("inputs_start")

    def inputs_end(self) -> None:
        self._mark("inputs_end")

    def forward_end(self) -> None:
        if self._step is not None:
            self._step.forward_end.record()

    def drafter_start(self) -> None:
        if self._step is not None:
            self._step.drafter_start.record()

    def drafter_end(self) -> None:
        """Ends the step: only steps that reach here have a draft cost."""
        if self._step is not None:
            self._step.drafter_end.record()
            self._timed.append((self._step, self._batch))
            self._step = None
            if self._stream_every and len(self._timed) >= self._stream_every:
                self._resolve_stream()


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        check_ep_fault: bool = False,
        routed_experts: RoutedExpertsTensors | None = None,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.routed_experts = routed_experts
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)
        self._has_fault: torch.Tensor | None = None
        self._main_stream = main_stream
        self._copy_stream = copy_stream
        self._draft_copy_event: torch.cuda.Event | None = None
        self._draft_token_ids: np.ndarray | None = None
        self._draft_req_ids: list[str] = []
        self._draft_req_indices: list[int] = []
        self._draft_producer_step_id: int | None = None

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.sampling_mask_tensors: SamplingMaskTensors | None = None
            if sampler_output.sampling_mask_tensors is not None:
                self.sampling_mask_tensors = (
                    sampler_output.sampling_mask_tensors.to_cpu_nonblocking()
                )
            self.routed_experts_cpu: RoutedExpertsTensors | None = None
            if routed_experts is not None:
                self.routed_experts_cpu = routed_experts.to_cpu_nonblocking()
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            self.copy_event.record(copy_stream)

    def set_draft_token_ids(
        self,
        req_ids: list[str],
        draft_token_ids: torch.Tensor,
        structured_output_request_ids: list[str],
        producer_step_id: int,
    ) -> None:
        req_id_to_index = {req_id: i for i, req_id in enumerate(req_ids)}
        self._draft_req_ids = [
            req_id
            for req_id in structured_output_request_ids
            if req_id in req_id_to_index
        ]
        if not self._draft_req_ids:
            return

        self._draft_req_indices = [
            req_id_to_index[req_id] for req_id in self._draft_req_ids
        ]
        self._draft_producer_step_id = producer_step_id
        self._draft_copy_event = torch.cuda.Event(blocking=True)
        with stream(self._copy_stream, self._main_stream):
            self._copy_stream.wait_stream(self._main_stream)
            self._draft_token_ids = async_copy_to_np(draft_token_ids)
            draft_token_ids.record_stream(self._copy_stream)
            self._draft_copy_event.record(self._copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()
        if self._draft_copy_event is not None:
            self._draft_copy_event.synchronize()
            assert self._draft_token_ids is not None
            assert self._draft_producer_step_id is not None
            self.model_runner_output.draft_token_ids = DraftTokenIds(
                req_ids=self._draft_req_ids,
                draft_token_ids=self._draft_token_ids[self._draft_req_indices].tolist(),
                producer_step_id=self._draft_producer_step_id,
            )

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.sampling_mask_tensors is not None:
            self.model_runner_output.sampling_masks = (
                self.sampling_mask_tensors.tolists(self.num_sampled_tokens_np)
            )

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )
            if envs.VLLM_RAISE_ON_LOGIT_NANS:
                raise_if_nan_logits(self.model_runner_output.num_nans_in_logits)

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        if self.routed_experts_cpu is not None:
            self.model_runner_output.routed_experts = self.routed_experts_cpu.tolists()

        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: "
                "one or more ranks timed out during dispatch/combine. "
                f"Mask: {mask.cpu().tolist()}"
            )

        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: PoolerOutput,
        finished_mask: list[bool],
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            if isinstance(self.pooler_output, torch.Tensor) and all(finished_mask):
                self.pooler_output_cpu: PoolerOutput = self.pooler_output.to(
                    "cpu", non_blocking=True
                )
            else:
                outputs = (
                    self.pooler_output.unbind()
                    if isinstance(self.pooler_output, torch.Tensor)
                    else self.pooler_output
                )
                self.pooler_output_cpu = [
                    None
                    if output is None or not is_finished
                    else output.to("cpu", non_blocking=True)
                    for output, is_finished in zip(outputs, finished_mask, strict=True)
                ]
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        if isinstance(self.pooler_output_cpu, torch.Tensor):
            pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        else:
            pooler_output = self.pooler_output_cpu
        self.copy_event.synchronize()
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
