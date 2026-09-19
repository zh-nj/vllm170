# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bindable sink for row-level prefill diagnostics.

``DeepseekV4Attention`` reports a few intermediate tensors per forward to
whatever sink the model installed at startup. That is how the prefill
batch-dependence investigation compares two chunk shapes stage by stage: the
first reported stage whose value differs is the one that introduces the
difference, and everything before it is innocent.

Nothing is bound in a normal run, so ``capture`` costs one attribute check and
the callers need no configuration of their own.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# Receives one named tensor per stage of an armed probe forward.
CaptureSink = Callable[[int | str, torch.Tensor, torch.Tensor], None]

_sink: CaptureSink | None = None


def bind(sink: CaptureSink | None) -> None:
    """Install or clear the sink. Only the model's probe dump calls this."""
    global _sink
    _sink = sink


def active() -> bool:
    return _sink is not None


def capture(key: str, tensor: torch.Tensor, positions: torch.Tensor) -> None:
    """Report ``tensor`` for the batch whose absolute positions are ``positions``."""
    sink = _sink
    if sink is not None:
        sink(key, tensor, positions)
