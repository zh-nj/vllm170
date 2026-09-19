# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CUDA-event profiling for DeepSeek-V4 serving phases."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Literal, TypeVar

import torch

from vllm.logger import init_logger
from vllm.profiler.phase_profile_hook import register_phase_profile_stage

logger = init_logger(__name__)

ProfileFlow = Literal["prefill", "decode", "dspark", "dflash"]
ProfileLevel = Literal["total", "module", "stage", "detail"]
_F = TypeVar("_F", bound=Callable[..., Any])

_PROFILE_ENABLED = os.getenv("VLLM_DEEPSEEK_V4_PHASE_PROFILE", "0") == "1"
_PROFILE_RAW_PATH = os.getenv("VLLM_DEEPSEEK_V4_PHASE_PROFILE_PATH", "")
_PROFILE_CONTROL_PATH = os.getenv("VLLM_DEEPSEEK_V4_PHASE_PROFILE_CONTROL_PATH", "")
_PROFILE_RECORD_FUNCTION = (
    os.getenv("VLLM_DEEPSEEK_V4_PHASE_PROFILE_RECORD_FUNCTION", "1") == "1"
)
_PROFILE_FLOWS = {
    flow.strip()
    for flow in os.getenv(
        "VLLM_DEEPSEEK_V4_PHASE_PROFILE_FLOWS", "prefill,decode,dspark,dflash"
    ).split(",")
    if flow.strip()
}
_PROFILE_LAYERS = {
    layer.strip()
    for layer in os.getenv("VLLM_DEEPSEEK_V4_PHASE_PROFILE_LAYERS", "").split(",")
    if layer.strip()
}
_PROFILE_STAGES = {
    stage.strip()
    for stage in os.getenv("VLLM_DEEPSEEK_V4_PHASE_PROFILE_STAGES", "").split(",")
    if stage.strip()
}


@dataclass(frozen=True)
class _PendingEvent:
    start: torch.cuda.Event
    end: torch.cuda.Event
    row: dict[str, Any]


class _ProfileContext(threading.local):
    def __init__(self) -> None:
        self.flow: ProfileFlow | None = None
        self.step_id: int | None = None
        self.token_count: int | None = None
        self.execution: str | None = None
        self.step_kind: str | None = None
        self.layer: str | None = None
        self.compress_ratio: int | None = None
        self.queue: list[_PendingEvent] = []
        self.next_step_id = 0
        self.stage_depth = 0


_CONTEXT = _ProfileContext()
_WRITE_LOCK = threading.Lock()


def phase_profile_enabled(flow: ProfileFlow | None = None) -> bool:
    """Return whether the opt-in measurement window accepts ``flow``."""
    if not _PROFILE_ENABLED or not _PROFILE_RAW_PATH:
        return False
    if flow is not None and flow not in _PROFILE_FLOWS:
        return False
    return not _PROFILE_CONTROL_PATH or Path(_PROFILE_CONTROL_PATH).exists()


def _is_compiling_or_capturing() -> bool:
    try:
        if torch.compiler.is_compiling():
            return True
    except (AttributeError, RuntimeError):
        pass
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


def _rank_metadata() -> tuple[int | None, int | None]:
    rank = None
    pp_rank = None
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            rank = dist.get_rank()
    except RuntimeError:
        pass
    try:
        from vllm.distributed.parallel_state import get_pp_group

        pp_rank = get_pp_group().rank_in_group
    except (AssertionError, RuntimeError):
        pass
    return rank, pp_rank


def _trace_path() -> Path:
    rank, pp_rank = _rank_metadata()
    try:
        cuda_device: str | int = torch.accelerator.current_device_index()
    except RuntimeError:
        cuda_device = "unknown"
    rendered = (
        _PROFILE_RAW_PATH.replace("{pid}", str(os.getpid()))
        .replace("{cuda_device}", str(cuda_device))
        .replace("{rank}", "unknown" if rank is None else str(rank))
        .replace("{pp_rank}", "unknown" if pp_rank is None else str(pp_rank))
    )
    return Path(rendered)


def _write_record(row: dict[str, Any]) -> None:
    rank, pp_rank = _rank_metadata()
    try:
        cuda_device = torch.accelerator.current_device_index()
    except RuntimeError:
        cuda_device = None
    record = {
        "schema_version": 2,
        "timestamp_ns": time.time_ns(),
        "pid": os.getpid(),
        "rank": rank,
        "pp_rank": pp_rank,
        "cuda_device": cuda_device,
        **row,
    }
    path = _trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def flush_phase_profile() -> None:
    """Resolve queued CUDA events with one device synchronization."""
    queue = _CONTEXT.queue
    if not queue:
        return
    try:
        torch.accelerator.synchronize()
    except RuntimeError:
        logger.warning(
            "DeepSeek-V4 phase profile synchronization failed", exc_info=True
        )
        queue.clear()
        return
    pending, _CONTEXT.queue = queue, []
    for event in pending:
        try:
            elapsed_us = event.start.elapsed_time(event.end) * 1000.0
        except RuntimeError:
            logger.warning(
                "DeepSeek-V4 phase profile event failed for %s",
                event.row["phase"],
                exc_info=True,
            )
            continue
        row = _materialize_cuda_scalars(event.row)
        _write_record({**row, "elapsed_us": elapsed_us})


def _materialize_cuda_scalars(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("phase profile metadata tensors must be scalar")
        return value.item()
    if isinstance(value, dict):
        return {key: _materialize_cuda_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize_cuda_scalars(item) for item in value]
    return value


@contextmanager
def phase_profile_step(
    flow: ProfileFlow,
    *,
    token_count: int | None = None,
    execution: str | None = None,
    step_kind: str | None = None,
) -> Iterator[None]:
    """Attach flow metadata and flush all nested events at the step boundary."""
    if not phase_profile_enabled(flow) or _is_compiling_or_capturing():
        yield
        return
    context = _CONTEXT
    previous = (
        context.flow,
        context.step_id,
        context.token_count,
        context.execution,
        context.step_kind,
    )
    context.flow = flow
    context.step_id = context.next_step_id
    context.next_step_id += 1
    context.token_count = token_count
    context.execution = execution
    context.step_kind = step_kind
    step_started_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        step_host_us = (time.perf_counter_ns() - step_started_ns) / 1000.0
        flush_phase_profile()
        _write_record(
            {
                "flow": flow,
                "stage": "step.host_total",
                "phase": f"dsv4.{flow}.step.host_total",
                "level": "total",
                "step_id": context.step_id,
                "token_count": token_count,
                "execution": execution,
                "step_kind": step_kind,
                "layer": None,
                "compress_ratio": None,
                "cpu_elapsed_us": step_host_us,
                "cpu_start_ns": step_started_ns,
                "elapsed_us": 0.0,
            }
        )
        (
            context.flow,
            context.step_id,
            context.token_count,
            context.execution,
            context.step_kind,
        ) = previous


@contextmanager
def phase_profile_layer(
    layer: str,
    *,
    compress_ratio: int | None = None,
) -> Iterator[None]:
    """Attach the current DSV4 layer to nested phase records."""
    if not _PROFILE_ENABLED:
        yield
        return
    previous = (_CONTEXT.layer, _CONTEXT.compress_ratio)
    _CONTEXT.layer = layer
    _CONTEXT.compress_ratio = compress_ratio
    try:
        yield
    finally:
        _CONTEXT.layer, _CONTEXT.compress_ratio = previous


def phase_profile_stage(
    stage: str,
    ref: torch.Tensor,
    *,
    flow: ProfileFlow | None = None,
    level: ProfileLevel = "detail",
    extra: dict[str, Any] | None = None,
):
    """Measure one GPU stage and expose the same label to Torch traces."""
    # Draft backbones can reuse target modules whose backend naturally calls
    # itself "decode". Keep nested records in the outer draft flow.
    actual_flow = (
        _CONTEXT.flow
        if _CONTEXT.flow in {"dspark", "dflash"}
        else flow or _CONTEXT.flow
    )
    if _PROFILE_STAGES and stage not in _PROFILE_STAGES:
        return nullcontext()
    current_layer = _CONTEXT.layer
    if (
        _PROFILE_LAYERS
        and current_layer is not None
        and current_layer.removesuffix(".attn") not in _PROFILE_LAYERS
    ):
        return nullcontext()
    if (
        actual_flow is None
        or not phase_profile_enabled(actual_flow)
        or not ref.is_cuda
        or _is_compiling_or_capturing()
    ):
        return nullcontext()

    label = f"dsv4.{actual_flow}.{stage}"

    @contextmanager
    def _scope() -> Iterator[None]:
        context = _CONTEXT
        previous_flow = context.flow
        previous_step_kind = context.step_kind
        context.flow = actual_flow
        if context.step_kind is None and extra:
            extra_step_kind = extra.get("draft_kind", extra.get("dspark_kind"))
            if isinstance(extra_step_kind, str):
                context.step_kind = extra_step_kind
        context.stage_depth += 1
        record_scope = (
            torch.autograd.profiler.record_function(label)
            if _PROFILE_RECORD_FUNCTION
            else nullcontext()
        )
        cpu_started_ns = time.perf_counter_ns()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        try:
            with record_scope:
                start.record()
                try:
                    yield
                finally:
                    end.record()
            row: dict[str, Any] = {
                "flow": actual_flow,
                "stage": stage,
                "phase": label,
                "level": level,
                "step_id": context.step_id,
                "token_count": context.token_count,
                "execution": context.execution,
                "step_kind": context.step_kind,
                "layer": context.layer,
                "compress_ratio": context.compress_ratio,
                "cpu_elapsed_us": (time.perf_counter_ns() - cpu_started_ns) / 1000.0,
                "cpu_start_ns": cpu_started_ns,
            }
            if extra:
                row["extra"] = extra
            context.queue.append(_PendingEvent(start, end, row))
        finally:
            context.stage_depth -= 1
            context.flow = previous_flow
            context.step_kind = previous_step_kind
        if context.step_id is None and context.stage_depth == 0:
            flush_phase_profile()

    return _scope()


register_phase_profile_stage(phase_profile_stage)


def classify_target_flow(is_prefilling: Any) -> ProfileFlow:
    """Classify a target batch; mixed batches are attributed to prefill."""
    try:
        return "prefill" if bool(is_prefilling.any()) else "decode"
    except (AttributeError, TypeError):
        return "decode"


def classify_speculator_flow(speculator_name: Any) -> ProfileFlow | None:
    """Map a supported speculator to its profiling flow."""
    flows: dict[str, ProfileFlow] = {
        "DFlash": "dflash",
        "DFlash2": "dflash",
        "DSpark": "dspark",
    }
    return flows.get(speculator_name) if isinstance(speculator_name, str) else None


def profile_parallel_draft_step(function: _F) -> _F:
    """Attach one DFlash or DSpark proposal to its draft profiling flow."""

    @wraps(function)
    def wrapped(self: Any, input_batch: Any, *args: Any, **kwargs: Any) -> Any:
        speculator_name = getattr(self, "_speculator_name", None)
        flow = classify_speculator_flow(speculator_name)
        if flow is None:
            return function(self, input_batch, *args, **kwargs)
        if getattr(self, "has_draft_role", True) is False:
            # Ranks that own no piece of the draft step, e.g. PP0 in a [1,2,3]
            # DSpark layout, return immediately; wrapping them only puts an empty
            # row in the draft ledger and costs an event pair per step.
            return function(self, input_batch, *args, **kwargs)
        graph_mode = self.vllm_config.compilation_config.cudagraph_mode.decode_mode()
        execution = (
            "eager"
            if kwargs.get("is_profile", False) or graph_mode.value == 0
            else "graph_or_eager"
        )
        step_kind = classify_target_flow(input_batch.is_prefilling_np)
        with (
            phase_profile_step(
                flow,
                token_count=input_batch.num_tokens,
                execution=execution,
                step_kind=step_kind,
            ),
            phase_profile_stage(
                "draft.propose_total",
                input_batch.input_ids,
                level="total",
                extra={
                    "draft_kind": step_kind,
                    "speculator": speculator_name,
                    "num_reqs": input_batch.num_reqs,
                    "num_draft_tokens": input_batch.num_draft_tokens,
                    "num_query_tokens": (input_batch.num_reqs * self.num_query_per_req),
                },
            ),
        ):
            return function(self, input_batch, *args, **kwargs)

    return wrapped  # type: ignore[return-value]
