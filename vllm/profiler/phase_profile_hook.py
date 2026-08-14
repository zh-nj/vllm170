# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional hook for model-specific CUDA-event phase profiling."""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

PhaseProfileStage = Callable[..., AbstractContextManager[None]]

_phase_profile_stage: PhaseProfileStage | None = None


def register_phase_profile_stage(callback: PhaseProfileStage) -> None:
    """Register the active model-specific phase profiler."""
    global _phase_profile_stage
    _phase_profile_stage = callback


def optional_phase_profile_stage(
    stage: str,
    ref: torch.Tensor,
    *,
    level: str = "detail",
    extra: dict[str, Any] | None = None,
) -> AbstractContextManager[None]:
    """Call the registered profiler or return a no-op context."""
    if _phase_profile_stage is None:
        return nullcontext()
    return _phase_profile_stage(stage, ref, level=level, extra=extra)
