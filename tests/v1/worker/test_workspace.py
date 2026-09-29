# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import pytest
import torch

import vllm.v1.worker.workspace as workspace
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import _num_workspace_lanes


class _SpecConfig:
    def __init__(self, dspark: bool) -> None:
        self._dspark = dspark

    def use_dspark(self) -> bool:
        return self._dspark


class _VllmConfig:
    def __init__(self, spec_config: _SpecConfig | None) -> None:
        self.speculative_config = spec_config


@pytest.mark.parametrize(
    ("use_v2_model_runner", "spec_config", "expected"),
    [
        (True, _SpecConfig(True), 3),
        (True, _SpecConfig(False), 3),
        (True, None, 3),
        (False, _SpecConfig(True), 3),
        (False, None, 3),
    ],
)
def test_workspace_lane_count_covers_every_owner(
    use_v2_model_runner: bool,
    spec_config: _SpecConfig | None,
    expected: int,
) -> None:
    """The V2 runner configures every lane its owners can select.

    The count no longer moves with DSpark: the eager-only prefill lane sits above
    the draft lane, so it covers both, and slot allocation stays lazy.
    """
    config = cast(VllmConfig, _VllmConfig(spec_config))
    assert _num_workspace_lanes(config, use_v2_model_runner) == expected


def test_prefill_workspace_lane_is_configured_for_the_v2_runner() -> None:
    """Selecting an unconfigured lane raises, so the count must cover it."""
    lanes = _num_workspace_lanes(cast(VllmConfig, _VllmConfig(_SpecConfig(True))), True)
    assert lanes > workspace.PREFILL_WORKSPACE_LANE


def test_workspace_lanes_do_not_alias_and_restore_context(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    assert manager._current_workspaces == [None, None, None, None]

    (target,) = manager.get_simultaneous(((512,), torch.uint8))
    with workspace.use_workspace_lane(1):
        (draft,) = manager.get_simultaneous(((256,), torch.uint8))
        (draft_reused,) = manager.get_simultaneous(((8,), torch.uint8))
    (target_reused,) = manager.get_simultaneous(((8,), torch.uint8))

    assert manager._current_workspaces[0].numel() == 512  # type: ignore[union-attr]
    assert manager._current_workspaces[1].numel() == 256  # type: ignore[union-attr]
    assert manager._current_workspaces[2:] == [None, None]
    assert target.data_ptr() != draft.data_ptr()
    assert draft.data_ptr() == draft_reused.data_ptr()
    assert target.data_ptr() == target_reused.data_ptr()


def test_workspace_lanes_compose_with_ubatches(monkeypatch) -> None:
    active_ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch[0])
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    pointers = set()
    for ubatch_id in range(2):
        active_ubatch[0] = ubatch_id
        for lane in range(2):
            with workspace.use_workspace_lane(lane):
                (buffer,) = manager.get_simultaneous(((16,), torch.uint8))
                pointers.add(buffer.data_ptr())

    assert len(pointers) == 4


def test_workspace_lane_validation(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)

    with (
        pytest.raises(ValueError, match="non-negative"),
        workspace.use_workspace_lane(-1),
    ):
        pass

    with (
        workspace.use_workspace_lane(1),
        pytest.raises(RuntimeError, match="is not configured"),
    ):
        manager.get_simultaneous(((1,), torch.uint8))

    with pytest.raises(ValueError, match="at least one"):
        workspace.WorkspaceManager(torch.device("cpu"), num_lanes=0)


def test_prefill_gather_plan_enforces_the_fixed_budget() -> None:
    """The prefill gather budget must admit, split, or fall back — never grow.

    The budget is a hard ceiling: a context that does not fit one row is served
    by the direct paged-cache reader, and a batch that fits fewer rows is split
    into more chunks instead of allocating more.
    """
    from vllm.models.deepseek_v4_1.amd.rocm import (
        _prefill_gather_plan,
        _prefill_gather_width,
    )

    head_dim, element_bytes, window, batched = 512, 2, 128, 2048
    max_rows = 4

    def plan(context_rows: int, num_prefills: int, budget_tokens: int):
        budget = (
            _prefill_gather_width(budget_tokens, window, batched)
            * head_dim
            * element_bytes
            if budget_tokens > 0
            else 0
        )
        return _prefill_gather_plan(
            context_rows,
            num_prefills,
            window,
            batched,
            head_dim,
            element_bytes,
            budget,
            max_rows,
        )

    # Unbounded keeps the historical "one row per concurrent prefill" behaviour.
    assert plan(1 << 20, 8, 0) == (max_rows, False)

    # A 64k-token budget holds one 32k-context row, so four concurrent prefills
    # are served one row at a time instead of allocating four.
    assert plan(32768, 4, 65536) == (1, False)

    # Room for four 32k-context rows: no chunking needed.
    assert plan(32768, 4, 262144) == (max_rows, False)

    # The same 64k budget against a 64k-context row still fits exactly one row.
    assert plan(65536, 4, 65536) == (1, False)

    # A context wider than the whole budget cannot be gathered at all: the batch
    # is admitted to the direct paged-cache reader instead.
    rows, admit = plan(1 << 20, 1, 65536)
    assert admit and rows == 1, (rows, admit)

    # Below one row the gather is refused; above it the row count grows with the
    # budget and never shrinks, saturating at the per-chunk cap.
    fitted = []
    for budget in (32768, 65536, 131072, 262144, 1 << 20):
        rows, admit = plan(65536, 4, budget)
        if admit:
            assert rows == 1
            continue
        fitted.append(rows)
    assert fitted == sorted(fitted), fitted
    assert fitted[-1] == max_rows, fitted


def test_persistent_resources_are_per_slot_fixed_shape(monkeypatch) -> None:
    """Persistent resources are created once per (ubatch, lane) and validated.

    They are the fixed-shape counterpart of the growable lane buffer: a late
    request for a different shape must fail loudly instead of reallocating, or a
    captured graph would keep pointing at the old storage.
    """
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=1, num_lanes=2
    )
    created = []

    def factory() -> torch.Tensor:
        created.append(1)
        return torch.zeros(4, dtype=torch.float32)

    first = manager.get_persistent_resource("scratch", factory)
    second = manager.get_persistent_resource("scratch", factory)
    assert first is second and len(created) == 1

    with workspace.use_workspace_lane(1):
        other = manager.get_persistent_resource("scratch", factory)
    assert other is not first and len(created) == 2

    tensor = manager.get_persistent("acc", (2, 3), torch.float32)
    assert tensor.shape == (2, 3) and tensor.dtype == torch.float32
    assert manager.get_persistent("acc", (2, 3), torch.float32) is tensor
    with pytest.raises(ValueError):
        manager.get_persistent("acc", (3, 2), torch.float32)


def test_locking_is_per_lane_and_blocks_new_persistent_keys(monkeypatch) -> None:
    """A capacity-fixed lane can be frozen without freezing the growable ones."""
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=1, num_lanes=3
    )
    with workspace.use_workspace_lane(2):
        (frozen,) = manager.get_simultaneous(((1024,), torch.uint8))
        manager.get_persistent("reserved", (8,), torch.float32)

    manager.lock(lanes=[2])
    assert manager.is_locked(2) and not manager.is_locked(0)

    # The frozen lane may reuse what it reserved, but must not grow or create a
    # key that warmup never allocated.
    with workspace.use_workspace_lane(2):
        (reused,) = manager.get_simultaneous(((512,), torch.uint8))
        assert reused.data_ptr() == frozen.data_ptr()
        with pytest.raises(AssertionError):
            manager.get_simultaneous(((2048,), torch.uint8))
        with pytest.raises(AssertionError):
            manager.get_persistent("late", (8,), torch.float32)

    # An unlocked lane still grows, and may still create new persistent keys:
    # the lock is per lane, not a global freeze.
    (grown,) = manager.get_simultaneous(((4096,), torch.uint8))
    assert grown.numel() == 4096
    late_but_allowed = manager.get_persistent("other-lane", (4,), torch.float32)
    assert late_but_allowed.shape == (4,)

    manager.unlock(lanes=[2])
    with workspace.use_workspace_lane(2):
        (regrown,) = manager.get_simultaneous(((4096,), torch.uint8))
        assert regrown.numel() == 4096


def test_prefill_gather_budget_folds_the_compress_ratio() -> None:
    """A compressed layer's budget must match the row its warmup reserved.

    The knob counts context tokens while a gather row holds compressed rows; a
    layer with compress_ratio=2 reserves and may gather half as wide a row, so a
    budget that skipped the division would admit a row the warmup never reserved.
    """
    from vllm.models.deepseek_v4_1.amd.rocm import (
        _prefill_gather_budget_bytes,
        _prefill_gather_width,
    )

    head_dim, element_bytes, window, batched = 512, 2, 128, 2048
    full = _prefill_gather_budget_bytes(
        65536, 1, window, batched, head_dim, element_bytes
    )
    halved = _prefill_gather_budget_bytes(
        65536, 2, window, batched, head_dim, element_bytes
    )
    assert halved == (
        _prefill_gather_width(32768, window, batched) * head_dim * element_bytes
    )
    assert halved < full
    assert (
        _prefill_gather_budget_bytes(0, 1, window, batched, head_dim, element_bytes)
        == 0
    )


def test_split_k_partials_are_created_at_warmup(monkeypatch) -> None:
    """Split-K partials must survive a lane lock taken after capture.

    The warmup and the serving path share one entry point, so a lock cannot meet
    a partial key the warmup never created (the lane's capacity reservation has to
    account for them too).
    """
    from vllm.models.deepseek_v4_1.amd.rocm import _prefill_split_k_partials

    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=1, num_lanes=3
    )

    with workspace.use_workspace_lane(2):
        warm_partials = _prefill_split_k_partials(manager, 16, 2, 4, 8)

    manager.lock(lanes=[2])

    with workspace.use_workspace_lane(2):
        served_partials = _prefill_split_k_partials(manager, 16, 2, 4, 8)

    for warm, served in zip(warm_partials, served_partials):
        assert warm is served, "serving must reuse the warmup allocation"
        assert warm.dtype == torch.float32
    assert warm_partials[2].shape == (16, 2, 4, 8)
