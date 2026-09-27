# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

from typing import Any

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.phase_profile import (
    phase_profile_enabled,
    phase_profile_stage,
    profile_parallel_draft_step,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    prepare_dflash_inputs,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    dspark_backbone_vllm_config,
    load_dspark_model,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)


def _relay_profile_extra(
    package: dict[str, Any],
) -> dict[str, Any] | None:
    """Describe one DSpark relay package when the opt-in profile is active."""
    if not phase_profile_enabled("dspark"):
        return None

    tensor_keys: list[str] = []
    non_tensor_keys: list[str] = []
    tensor_bytes = 0
    bytes_by_dtype: dict[str, int] = {}
    for key, value in package.items():
        if not isinstance(value, torch.Tensor):
            non_tensor_keys.append(key)
            continue
        tensor_keys.append(key)
        size = value.numel() * value.element_size()
        tensor_bytes += size
        dtype_name = str(value.dtype)
        bytes_by_dtype[dtype_name] = bytes_by_dtype.get(dtype_name, 0) + size
    return {
        "tensor_count": len(tensor_keys),
        "tensor_bytes": tensor_bytes,
        "tensor_bytes_by_dtype": bytes_by_dtype,
        "tensor_keys": tensor_keys,
        "non_tensor_count": len(non_tensor_keys),
        "non_tensor_keys": non_tensor_keys,
    }


def _send_relay_package(
    group: Any,
    package: dict[str, Any],
    *,
    dst: int,
    ref: torch.Tensor,
    stage: str,
) -> None:
    """Split a profiled relay send into launch and handle-wait phases."""
    if not phase_profile_enabled("dspark"):
        group.send_tensor_dict(package, dst=dst)
        return

    with phase_profile_stage(f"{stage}.launch", ref, level="detail"):
        handles = group.isend_tensor_dict(package, dst=dst)
    if handles:
        with phase_profile_stage(f"{stage}.metadata_wait", ref, level="detail"):
            handles[0].wait()
        with phase_profile_stage(f"{stage}.tensor_wait", ref, level="detail"):
            for handle in handles[1:]:
                handle.wait()

    pending = getattr(group, "_pending_isends", None)
    if pending and pending[-1][0] is handles:
        pending.pop()


class _DSparkStageStaging:
    """Static buffers for the tensors one DSpark step reads from and writes to.

    Each participating rank exchanges its piece of the draft state over the PP
    relay, but a FULL cudagraph can only read fixed addresses: the relayed
    tensors are copied in before a replay and copied out after it. Everything is
    sized for one draft step, i.e. ``max_num_reqs * num_query_per_req`` rows.
    """

    def __init__(
        self,
        max_rows: int,
        hidden_size: int,
        hc_mult: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.max_rows = max_rows
        shapes = {
            "hidden": (max_rows, hidden_size),
            "residual": (max_rows, hc_mult, hidden_size),
        }
        mix_shapes = {
            "post_mix": (max_rows, hc_mult, 1),
            "res_mix": (max_rows, hc_mult, hc_mult),
            "pre_mix": (max_rows, hc_mult),
        }
        self.inputs: dict[str, torch.Tensor] = {
            name: torch.empty(shape, dtype=dtype, device=device)
            for name, shape in shapes.items()
        }
        self.outputs: dict[str, torch.Tensor] = {
            name: torch.empty(shape, dtype=dtype, device=device)
            for name, shape in shapes.items()
        }
        for name, shape in mix_shapes.items():
            self.inputs[name] = torch.empty(shape, dtype=torch.float32, device=device)
            self.outputs[name] = torch.empty(shape, dtype=torch.float32, device=device)

    def copy_in(self, rows: int, **tensors: torch.Tensor | None) -> None:
        """Stage the relayed tensors; ``None`` means the sender had no value."""
        for name, tensor in tensors.items():
            if tensor is None:
                continue
            self.inputs[name][:rows].copy_(tensor[:rows])

    def out(self, name: str, rows: int) -> torch.Tensor:
        return self.outputs[name][:rows]


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps
        query_tokens = self.speculative_config.dspark_num_query_tokens
        if query_tokens is not None:
            if not self.sample_from_anchor:
                raise ValueError(
                    "A separate DSpark query length requires sample_from_anchor=True"
                )
            if query_tokens * self.max_num_reqs > self.max_num_tokens:
                raise ValueError("DSpark queries exceed the token buffer capacity")
            trained = getattr(
                self.draft_model_config.hf_config, "dspark_block_size", query_tokens
            )
            if query_tokens > trained:
                if self.draft_model_config.hf_config.model_type != "deepseek_v41":
                    raise ValueError("DSpark queries exceed the checkpoint block size")
                logger.info(
                    "DeepSeek V4.1 DSpark uses %d query tokens "
                    "(checkpoint block size: %d)",
                    query_tokens,
                    trained,
                )
            self.num_query_per_req = query_tokens

        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )
        self._target_hidden_states_workspace: torch.Tensor | None = None

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )

        self.draft_token_confidence_probs = torch.empty_like(
            self.draft_tokens, dtype=torch.float32
        )
        self.enable_adaptive_verification = (
            self.speculative_config.enable_adaptive_verification
        )
        self._reference_dump_path = envs.VLLM_DEEPSEEK_V4_DSPARK_REFERENCE_DUMP_PATH
        self._reference_dump_active = False
        self._reference_dump_written = False
        self._reference_dump: dict[str, Any] | None = None

        self.distributed = self.speculative_config.use_distributed_dspark()
        self._distributed_step_id = -1
        self._compact_v41_relay = False
        self._prepare_ahead = False
        if (
            getattr(vllm_config.model_config.hf_config, "model_type", None)
            == "deepseek_v41"
        ):
            from vllm.models.deepseek_v4_1.dspark_relay import (
                compact_relay_enabled,
                prepare_ahead_enabled,
            )

            self._compact_v41_relay = compact_relay_enabled(vllm_config)
            self._prepare_ahead = prepare_ahead_enabled(vllm_config)
        if self._compact_v41_relay:
            logger.info("DSV41F DSpark compact relay enabled for pure decode steps.")
        if self._prepare_ahead:
            logger.info(
                "DSV41F DSpark prepare-ahead relay enabled for pure decode steps."
            )
        if self.distributed:
            self._pp_group = get_pp_group()
            stage_ranks = self.speculative_config.dspark_pipeline_stage_ranks
            assert stage_ranks is not None
            self._stage_ranks = tuple(stage_ranks)
            self._pp_rank = self._pp_group.rank_in_group
            self._head_rank = self._pp_group.world_size - 1
            self._stage_index = (
                self._stage_ranks.index(self._pp_rank)
                if self._pp_rank in self._stage_ranks
                else None
            )
        else:
            self._stage_ranks = ()
            self._pp_rank = 0
            self._head_rank = 0
            self._stage_index = None

        # Distributed DSpark runs one rank-local piece of the draft step per
        # participating rank: the stage ranks run their share of the draft
        # blocks, the head rank reconstructs the collapsed draft state and runs
        # the sequential Markov head. Both read tensors that arrive over the PP
        # relay, and a FULL cudagraph can only read fixed addresses, so the
        # relayed tensors are staged into the buffers below before each replay.
        # Row count is bounded by one draft step: max_num_reqs requests of
        # num_query_per_req query tokens each. Only ranks that own a draft piece
        # need it: in a [1,2,3] layout PP0 owns nothing and never replays.
        self._graph_replay_logged = False
        self.staging = (
            _DSparkStageStaging(
                max_rows=self.max_num_reqs * self.num_query_per_req,
                hidden_size=self.draft_model_config.get_hidden_size(),
                hc_mult=self.draft_model_config.hf_config.hc_mult,
                dtype=self.dtype,
                device=device,
            )
            if self._has_draft_stage() or self._is_draft_head()
            else None
        )

    @property
    def has_draft_role(self) -> bool:
        """Whether this rank runs any part of the draft step at all.

        PP ranks outside ``dspark_pipeline_stage_ranks`` and the head -- PP0 in a
        ``[1,2,3]`` layout -- own no draft block, no draft weights and no draft
        KV cache, so their per-step draft call returns on its first line. Callers
        and instrumentation use this to skip them rather than measure a no-op.
        """
        if not self.distributed:
            return True
        return self._has_draft_stage() or self._is_draft_head()

    def _has_draft_stage(self) -> bool:
        return self.distributed and self._stage_index is not None

    def _is_draft_head(self) -> bool:
        return (
            self.distributed
            and self._stage_index is None
            and self._pp_rank == self._head_rank
        )

    def _capture_reference_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if not self._reference_dump_active:
            return
        assert self._reference_dump is not None
        self._reference_dump[name] = tensor.detach().to(device="cpu", copy=True)

    def set_target_hidden_states_workspace(self, workspace: torch.Tensor) -> bool:
        if (
            workspace.is_contiguous()
            and workspace.dtype == self.dtype
            and workspace.device == self.device
        ):
            self._target_hidden_states_workspace = workspace.view(-1)
            self.hidden_states = None
            return True
        return False

    def _prepare_context_hidden_states(
        self,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_target_tokens: int,
    ) -> torch.Tensor:
        if self._target_hidden_states_workspace is None:
            hidden_states = super()._prepare_context_hidden_states(
                last_hidden_states, aux_hidden_states, num_target_tokens
            )
            if aux_hidden_states:
                for i, tensor in enumerate(aux_hidden_states):
                    self._capture_reference_tensor(
                        f"target_aux_hidden_{i}", tensor[:num_target_tokens]
                    )
            self._capture_reference_tensor(
                "context_main_x", hidden_states[:num_target_tokens]
            )
            return hidden_states
        if not aux_hidden_states:
            hidden_states = last_hidden_states[:num_target_tokens]
            self._capture_reference_tensor("context_main_x", hidden_states)
            return hidden_states

        num_rows = aux_hidden_states[0].shape[0]
        packed_width = sum(tensor.shape[-1] for tensor in aux_hidden_states)
        packed_numel = num_rows * packed_width
        workspace = self._target_hidden_states_workspace
        can_reuse_workspace = (
            workspace is not None
            and workspace.numel() >= packed_numel
            and all(
                tensor.untyped_storage().data_ptr()
                != workspace.untyped_storage().data_ptr()
                for tensor in aux_hidden_states
            )
        )
        if can_reuse_workspace:
            assert workspace is not None
            packed_hidden_states = workspace[:packed_numel].view(num_rows, packed_width)
            torch.cat(aux_hidden_states, dim=-1, out=packed_hidden_states)
        else:
            packed_hidden_states = torch.cat(aux_hidden_states, dim=-1)

        hidden_states = self.model.combine_hidden_states(packed_hidden_states)
        for i, tensor in enumerate(aux_hidden_states):
            self._capture_reference_tensor(
                f"target_aux_hidden_{i}", tensor[:num_target_tokens]
            )
        self._capture_reference_tensor(
            "context_main_x", hidden_states[:num_target_tokens]
        )
        return hidden_states[:num_target_tokens]

    @property
    def attn_vllm_config(self) -> VllmConfig:
        return dspark_backbone_vllm_config(super().attn_vllm_config)

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        self._ced_context_source = getattr(target_model, "take_ced_draft_context", None)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        if self.enable_adaptive_verification and model.model.confidence_head is None:
            raise ValueError(
                "Adaptive verification needs a DSpark checkpoint with a confidence "
                "head, and this one has none. Pass "
                "enable_adaptive_verification=false in the speculative config to verify"
                " a fixed number of drafts instead."
            )
        return model

    def set_attn(self, *args: Any, **kwargs: Any) -> None:
        if not self.distributed or self.draft_attn_layer_names:
            super().set_attn(*args, **kwargs)
            return
        DraftModelSpeculator.set_attn(self, *args, **kwargs)
        self.draft_kv_cache_group_ids = []
        self.draft_kv_cache_group_id = -1
        self._context_slot_mappings = torch.empty(
            (0, self.max_num_tokens), dtype=torch.int64, device=self.device
        )
        self._layer_group_idx = []
        self._group_causal = not self.requires_non_causal

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if not self.distributed:
            super().init_cudagraph_manager(cudagraph_mode)
            return
        if not (self._has_draft_stage() or self._is_draft_head()):
            # Ranks that own no piece of the draft keep no draft cudagraphs.
            self.query_cudagraph_manager = None
            return
        wants_full = (
            not self.draft_model_config.enforce_eager
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        )
        supports_full = (
            self.attn_cg_support.min_cg_support.value
            >= AttentionCGSupport.UNIFORM_BATCH.value
        )
        if wants_full and not supports_full:
            logger.warning(
                "Distributed %s draft attention (%s) does not support full CUDA "
                "graphs; running the rank-local draft stage eagerly.",
                self._speculator_name,
                self.attn_cg_support.min_cg_attn_backend,
            )
        if wants_full and supports_full:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE
        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.num_query_per_req,
        )

    def capture(self) -> None:
        if not self.distributed:
            super().capture()
            return
        manager = self.query_cudagraph_manager
        if manager is None or manager.cudagraph_mode == CUDAGraphMode.NONE:
            return
        logger.info(
            "Capturing model for distributed %s speculator...", self._speculator_name
        )
        if self._is_draft_head():
            # Padded sample rows must not scatter into a live request during
            # capture; the real values are restored before every replay.
            self.sample_indices.zero_()
            self.sample_pos.zero_()
            self.sample_idx_mapping.fill_(-1)
        manager.capture(
            self._draft_stage_forward
            if self._has_draft_stage()
            else self._draft_head_forward,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=self._group_causal,
            progress_bar_desc=(
                f"Capturing distributed {self._speculator_name.lower()} CUDA graphs"
            ),
            skip_attn=not self._has_draft_stage(),
        )
        logger.info(
            "Distributed %s: captured %d CUDA graphs for the rank-local draft step.",
            self._speculator_name,
            len(manager.graphs),
        )

    def _note_graph_replay(self) -> None:
        # Not logger.info_once: that defaults to the local first rank scope, and
        # the ranks that own draft work are not the first rank.
        if self._graph_replay_logged:
            return
        self._graph_replay_logged = True
        logger.info(
            "Distributed %s is replaying captured CUDA graphs for the rank-local "
            "draft step.",
            self._speculator_name,
        )

    def _draft_stage_forward(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
    ) -> None:
        """The rank-local draft blocks, with the relayed state read from staging."""
        staging = self.staging
        assert staging is not None
        if self._stage_index == 0:
            # The first stage embeds its own query block from the static input
            # buffer, so only its outputs need staging.
            hidden_states = self.model.prepare_pipeline_query(
                self.input_buffers.input_ids[:num_tokens]
            )
            residual = post_mix = res_mix = pre_mix = None
        else:
            hidden_states = staging.inputs["hidden"][:num_tokens]
            residual = staging.inputs["residual"][:num_tokens]
            post_mix = staging.inputs["post_mix"][:num_tokens]
            res_mix = staging.inputs["res_mix"][:num_tokens]
            pre_mix = staging.inputs["pre_mix"][:num_tokens]

        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        ):
            hidden_states, residual, post_mix, res_mix, pre_mix = (
                self.model.forward_local_stage(
                    hidden_states,
                    self.input_buffers.positions[:num_tokens],
                    self.input_buffers.input_ids[:num_tokens],
                    residual,
                    post_mix,
                    res_mix,
                    pre_mix,
                )
            )

        staging.outputs["hidden"][:num_tokens].copy_(hidden_states)
        staging.outputs["residual"][:num_tokens].copy_(residual)
        staging.outputs["post_mix"][:num_tokens].copy_(post_mix)
        staging.outputs["res_mix"][:num_tokens].copy_(res_mix)
        staging.outputs["pre_mix"][:num_tokens].copy_(pre_mix)

    def _draft_head_forward(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
    ) -> None:
        """The head rank's step: collapse the draft state, then Markov sampling."""
        staging = self.staging
        assert staging is not None
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        ):
            head_hidden = self.model.finish_pipeline_query(
                staging.inputs["hidden"][:num_tokens],
                staging.inputs["residual"][:num_tokens],
                staging.inputs["post_mix"][:num_tokens],
                staging.inputs["res_mix"][:num_tokens],
                staging.inputs["pre_mix"][:num_tokens],
            )
            self._sample_sequential(num_reqs, head_hidden)

    def set_distributed_step_id(self, scheduler_step_id: int) -> None:
        self._distributed_step_id = scheduler_step_id

    def _validate_distributed_package(
        self, package: dict[str, Any], input_batch: Any
    ) -> None:
        if package["scheduler_step_id"] != self._distributed_step_id:
            raise RuntimeError(
                "Distributed DSpark scheduler step mismatch: "
                f"expected {self._distributed_step_id}, got "
                f"{package['scheduler_step_id']}."
            )
        is_dummy = package["dummy_run"]
        if is_dummy != (self._distributed_step_id == -1):
            raise RuntimeError("Distributed DSpark dummy-run marker mismatch.")
        if not is_dummy and package["req_ids"] != tuple(input_batch.req_ids):
            raise RuntimeError("Distributed DSpark request batch mismatch.")
        if package["num_reqs"] != input_batch.num_reqs:
            raise RuntimeError("Distributed DSpark request count mismatch.")

    def _uses_prepare_ahead(self, input_batch: Any, dummy_run: bool) -> bool:
        """Whether this step splits context/control from the layer states.

        Only a pure decode step can: a prefill-carrying batch keeps the single
        package, and a dummy step must stay on the path its capture uses.
        """
        return self._prepare_ahead and not input_batch.has_prefill and not dummy_run

    def _relay_prepare_package(
        self,
        package: dict[str, Any],
        input_batch: Any,
    ) -> None:
        """Forward the shared context/control package to the next draft stage.

        Nothing in it depends on this stage's draft output, so passing it on
        first is what lets the downstream stages prepare in parallel with this
        stage's query. The last stage has no one to forward to.
        """
        assert self._stage_index is not None
        next_index = self._stage_index + 1
        if next_index >= len(self._stage_ranks):
            return
        with phase_profile_stage(
            "draft.stage.prepare_send", input_batch.input_ids, level="stage"
        ):
            _send_relay_package(
                self._pp_group,
                package,
                dst=self._stage_ranks[next_index],
                ref=input_batch.input_ids,
                stage="draft.stage.prepare_send",
            )

    def _prepare_distributed_stage(
        self,
        input_batch: Any,
        package: dict[str, Any],
    ) -> tuple[int, dict[str, Any] | None, dict[str, torch.Tensor] | None]:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        self.draft_max_seq_len = min(
            int(input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item())
            + self.num_query_per_req,
            self.max_model_len,
        )
        if package["skip_attn"]:
            self.model.precompute_and_store_context_kv(
                package["context_main_x"],
                self.context_positions[:num_target_tokens],
            )
            return num_query_tokens, None, None

        with phase_profile_stage(
            "draft.stage.prepare_inputs",
            package["context_main_x"],
            level="stage",
        ):
            self._prepare_stage_inputs(input_batch, package)

        layer_group_idx = self._layer_group_idx or []
        context_slots = [
            self._context_slot_mappings[group_idx][:num_target_tokens]
            for group_idx in layer_group_idx
        ]
        with phase_profile_stage(
            "draft.context_kv", package["context_main_x"], level="stage"
        ):
            self.model.precompute_and_store_context_kv(
                package["context_main_x"],
                self.context_positions[:num_target_tokens],
                context_slots,
            )
        if "ced_context_main_x" in package:
            # Scheduler rows can contain CED padding; restore the real suffix
            # after that write so it also covers a final chunk shorter than SWA.
            self._restore_distributed_ced_context(package)
        with phase_profile_stage(
            "draft.stage.metadata", package["context_main_x"], level="stage"
        ):
            draft_attn_metadata = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs,
                num_tokens_padded=num_query_tokens,
                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
            slot_mappings = build_slot_mappings_by_layer(
                self.block_tables.slot_mappings[:, :num_query_tokens],
                self.kv_cache_config,
            )
        return num_query_tokens, draft_attn_metadata, slot_mappings

    def _restore_distributed_ced_context(self, package: dict[str, Any]) -> None:
        from vllm.models.deepseek_v4_1.dspark_relay import ced_context_slot_mappings

        positions = package["ced_context_positions"]
        with phase_profile_stage(
            "draft.ced_context_kv", package["ced_context_main_x"], level="stage"
        ):
            group_slots = ced_context_slot_mappings(
                positions,
                package["ced_context_req_indices"],
                self.block_tables,
                self.draft_kv_cache_group_ids,
            )
            assert self._layer_group_idx is not None
            self.model.precompute_and_store_context_kv(
                package["ced_context_main_x"],
                positions,
                [group_slots[index] for index in self._layer_group_idx],
            )
        if not getattr(self, "_ced_context_logged", False):
            self._ced_context_logged = True
            logger.info(
                "Distributed DSpark restored %d CED context rows on draft stage %d.",
                positions.numel(),
                self._stage_index,
            )

    def _prepare_stage_inputs(self, input_batch: Any, package: dict[str, Any]) -> None:
        for i, gid in enumerate(self.draft_kv_cache_group_ids):
            prepare_dflash_inputs(
                self.input_buffers,
                self.block_tables.slot_mappings[gid],
                self.context_positions,
                self._context_slot_mappings[i],
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                self.temperature,
                self.seeds,
                input_batch,
                package["num_sampled"],
                package["num_rejected"],
                package["last_sampled"],
                package["next_prefill_tokens"],
                package["temperature"],
                package["seeds"],
                self.block_tables.input_block_tables[gid],
                self.block_tables.kernel_block_sizes[gid],
                self.block_tables.cp_rank,
                self.block_tables.cp_size,
                self.block_tables.cp_interleave,
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                self.max_num_tokens,
                self.max_model_len,
                self.sample_from_anchor,
                request_idx_mapping=package["idx_mapping"],
            )

    @torch.inference_mode()
    @profile_parallel_draft_step
    def participate_distributed(self, input_batch: Any) -> None:
        if not self.distributed or self._stage_index is None:
            return
        src = (
            self._head_rank
            if self._stage_index == 0
            else self._stage_ranks[self._stage_index - 1]
        )
        with phase_profile_stage(
            "draft.stage.recv_wait", input_batch.input_ids, level="stage"
        ):
            package = self._pp_group.recv_tensor_dict(src=src)
        assert package is not None
        self._validate_distributed_package(package, input_batch)
        prepare_ahead = bool(package.get("prepare_ahead"))
        if prepare_ahead:
            # The context and sampling controls do not depend on the layer
            # below, so hand them on before doing any local work: every later
            # stage then prepares while this one still owes it a state.
            self._relay_prepare_package(package, input_batch)
        with phase_profile_stage(
            "draft.stage.prepare", package["context_main_x"], level="stage"
        ):
            num_query_tokens, attn_metadata, slot_mappings = (
                self._prepare_distributed_stage(input_batch, package)
            )
        if prepare_ahead and self._stage_index != 0:
            # The first draft stage builds its own query input; every later one
            # consumes the state this message carries, which was sent after the
            # prepare package on the same edge.
            with phase_profile_stage(
                "draft.stage.state_wait", input_batch.input_ids, level="stage"
            ):
                package = self._pp_group.recv_tensor_dict(src=src)
            assert package is not None
            if package.get("prepare_ahead"):
                raise RuntimeError(
                    "Distributed DSpark state package carries the prepare marker."
                )
            self._validate_distributed_package(package, input_batch)

        batch_desc, _ = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            input_batch.num_reqs,
            num_query_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            # Profile runs happen before the draft manager exists.
            need_eager=self.query_cudagraph_manager is None,
        )
        if batch_desc.cg_mode == CUDAGraphMode.FULL and not package["skip_attn"]:
            # `skip_attn` marks a memory-profiling run, whose block tables are
            # placeholders: the captured graph would write KV at real slots, so
            # those runs stay on the eager path. The whole rank-local draft step
            # is otherwise one graph; only the relayed state has to move in and
            # out of it.
            staging = self.staging
            manager = self.query_cudagraph_manager
            assert staging is not None
            assert manager is not None
            if self._stage_index != 0:
                staging.copy_in(
                    num_query_tokens,
                    hidden=package["hidden_states"],
                    residual=package["residual"],
                    post_mix=package["post_mix"],
                    res_mix=package["res_mix"],
                    pre_mix=package["pre_mix"],
                )
            self._note_graph_replay()
            with phase_profile_stage(
                "draft.stage.forward", input_batch.input_ids, level="stage"
            ):
                manager.run_fullgraph(batch_desc)
            hidden_states = staging.out("hidden", num_query_tokens)
            residual = staging.out("residual", num_query_tokens)
            post_mix = staging.out("post_mix", num_query_tokens)
            res_mix = staging.out("res_mix", num_query_tokens)
            pre_mix = staging.out("pre_mix", num_query_tokens)
        else:
            if self._stage_index == 0:
                hidden_states = self.model.prepare_pipeline_query(
                    self.input_buffers.input_ids[:num_query_tokens]
                )
                residual = post_mix = res_mix = pre_mix = None
            else:
                hidden_states = package["hidden_states"]
                residual = package["residual"]
                post_mix = package["post_mix"]
                res_mix = package["res_mix"]
                pre_mix = package["pre_mix"]

            batch_descriptor = BatchDescriptor(num_tokens=num_query_tokens)
            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_query_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=slot_mappings,
                batch_descriptor=batch_descriptor,
            ):
                hidden_states, residual, post_mix, res_mix, pre_mix = (
                    self.model.forward_local_stage(
                        hidden_states,
                        self.input_buffers.positions[:num_query_tokens],
                        self.input_buffers.input_ids[:num_query_tokens],
                        residual,
                        post_mix,
                        res_mix,
                        pre_mix,
                    )
                )

        package.update(
            hidden_states=hidden_states,
            residual=residual,
            post_mix=post_mix,
            res_mix=res_mix,
            pre_mix=pre_mix,
        )
        is_last_stage = self._stage_index == len(self._stage_ranks) - 1
        if (
            self._compact_v41_relay
            and not is_last_stage
            and not input_batch.has_prefill
            and not package["dummy_run"]
        ):
            # Only the head consumes these locally rebuilt buffers. Avoid
            # constructing unused views and discard any inbound copies.
            for key in (
                "query_input_ids",
                "sample_indices",
                "sample_pos",
                "sample_idx_mapping",
            ):
                package.pop(key, None)
        else:
            num_sample = input_batch.num_reqs * self.num_speculative_steps
            package.update(
                query_input_ids=self.input_buffers.input_ids[:num_query_tokens],
                sample_indices=self.sample_indices[:num_sample],
                sample_pos=self.sample_pos[:num_sample],
                sample_idx_mapping=self.sample_idx_mapping[:num_sample],
            )
        if is_last_stage:
            for key in (
                "context_main_x",
                "ced_context_main_x",
                "ced_context_positions",
                "ced_context_req_indices",
                "num_sampled",
                "num_rejected",
                "last_sampled",
                "next_prefill_tokens",
                "temperature",
                "seeds",
            ):
                package.pop(key, None)
        if prepare_ahead:
            # Under prepare-ahead only the states still depend on the layer
            # below; the context controls travelled in their own package, so
            # drop the copies this state package still carries. The marker goes
            # too: it describes the prepare package, not this one.
            for key in (
                "prepare_ahead",
                "context_main_x",
                "num_sampled",
                "num_rejected",
                "last_sampled",
                "next_prefill_tokens",
                "temperature",
                "seeds",
                "idx_mapping",
            ):
                package.pop(key, None)
        dst = (
            self._head_rank
            if is_last_stage
            else self._stage_ranks[self._stage_index + 1]
        )
        with phase_profile_stage(
            "draft.stage.send",
            self.input_buffers.input_ids,
            level="stage",
            extra=_relay_profile_extra(package),
        ):
            _send_relay_package(
                self._pp_group,
                package,
                dst=dst,
                ref=self.input_buffers.input_ids,
                stage="draft.stage.send",
            )

    def _take_ced_draft_context(self, input_batch: Any) -> Any:
        source = getattr(self, "_ced_context_source", None)
        context = source() if source is not None else None
        if context is not None:
            request_ids = context.request_ids or (context.request_id,)
            if not set(request_ids).issubset(input_batch.req_ids):
                raise ValueError("CED draft context belongs to another request")
        return context

    def _ced_draft_context_package(self, input_batch: Any) -> dict[str, torch.Tensor]:
        context = self._take_ced_draft_context(input_batch)
        if context is None:
            return {}
        request_ids = context.request_ids or (context.request_id,)
        lengths = context.request_lengths or (context.positions.numel(),)
        if (
            len(request_ids) != len(lengths)
            or sum(lengths) != context.positions.numel()
        ):
            raise ValueError("CED draft context request lengths do not match its rows")
        row_requests = [
            input_batch.req_ids.index(request_id)
            for request_id, length in zip(request_ids, lengths)
            for _ in range(length)
        ]
        return {
            "ced_context_main_x": self.model.combine_hidden_states(
                # Match the ordinary context workspace's projection dtype;
                # mHC replay can retain FP32 residual/auxiliary states.
                torch.cat(context.auxiliary, dim=-1).to(dtype=self.dtype)
            ),
            "ced_context_positions": context.positions,
            "ced_context_req_indices": torch.tensor(
                row_requests, dtype=torch.long, device=context.positions.device
            ),
        }

    def _publish_ced_draft_context(self, input_batch: Any) -> None:
        """Write the CED-replayed suffix into the draft KV cache before proposing.

        CED keeps a bounded per-request suffix of encoder hidden states and
        replays the decoder layers on it once per request; the replay produced
        exactly the states the drafter needs for its context KV, so publishing
        them here skips a redundant projection on the draft side.
        """
        context = self._take_ced_draft_context(input_batch)
        if context is None:
            return
        states = self.model.combine_hidden_states(torch.cat(context.auxiliary, dim=-1))
        assert self._layer_group_idx is not None
        slots = [
            context.slots[self.draft_kv_cache_group_ids[index]]
            for index in self._layer_group_idx
        ]
        self.model.precompute_and_store_context_kv(states, context.positions, slots)

    @torch.inference_mode()
    def propose(
        self,
        input_batch: Any,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: Any = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        if not dummy_run and not self.distributed:
            self._publish_ced_draft_context(input_batch)
        if not self.distributed:
            # ``DFlashSpeculator.propose`` carries its own phase-profile step.
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                dp_sync=dp_sync,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
            )
        return self._propose_distributed(
            input_batch,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
        )

    @profile_parallel_draft_step
    def _propose_distributed(
        self,
        input_batch: Any,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        *,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> torch.Tensor:
        if self._pp_rank != self._head_rank:
            raise RuntimeError("Only the final PP rank can initiate DSpark drafting.")
        if not aux_hidden_states:
            raise RuntimeError("Distributed DSpark requires target auxiliary states.")

        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        with phase_profile_stage(
            "draft.head.context_hidden", last_hidden_states, level="stage"
        ):
            context_main_x = self._prepare_context_hidden_states(
                last_hidden_states, aux_hidden_states, num_target_tokens
            )
        self._copy_request_inputs(num_reqs, input_batch.idx_mapping, temperature, seeds)
        # Pure decode steps may split the relay so the stages prepare in
        # parallel; prefill-carrying and dummy steps keep the single package.
        prepare_ahead = self._uses_prepare_ahead(input_batch, dummy_run)
        package: dict[str, Any] = {
            "scheduler_step_id": self._distributed_step_id,
            "dummy_run": dummy_run,
            "req_ids": tuple(input_batch.req_ids),
            "num_reqs": num_reqs,
            "idx_mapping": input_batch.idx_mapping[:num_reqs],
            "skip_attn": dummy_run and skip_attn_for_dummy_run,
            "prepare_ahead": prepare_ahead,
            "context_main_x": context_main_x,
            "num_sampled": num_sampled,
            "num_rejected": num_rejected,
            "last_sampled": last_sampled,
            "next_prefill_tokens": next_prefill_tokens,
            "temperature": temperature,
            "seeds": seeds,
        }
        if not dummy_run:
            package.update(self._ced_draft_context_package(input_batch))
        with phase_profile_stage(
            "draft.head.send",
            input_batch.input_ids,
            level="stage",
            extra=_relay_profile_extra(package),
        ):
            _send_relay_package(
                self._pp_group,
                package,
                dst=self._stage_ranks[0],
                ref=input_batch.input_ids,
                stage="draft.head.send",
            )
        with phase_profile_stage(
            "draft.head.recv_wait", input_batch.input_ids, level="stage"
        ):
            result = self._pp_group.recv_tensor_dict(src=self._stage_ranks[-1])
        assert result is not None
        self._validate_distributed_package(result, input_batch)

        with phase_profile_stage(
            "draft.head.result_copy", input_batch.input_ids, level="stage"
        ):
            self.input_buffers.input_ids[:num_query_tokens].copy_(
                result["query_input_ids"]
            )
            num_sample = num_reqs * self.num_speculative_steps
            self.sample_indices[:num_sample].copy_(result["sample_indices"])
            self.sample_pos[:num_sample].copy_(result["sample_pos"])
            self.sample_idx_mapping[:num_sample].copy_(result["sample_idx_mapping"])
        batch_desc, _ = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_query_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=self.query_cudagraph_manager is None,
        )
        self._prepare_eplb_forward(num_query_tokens)
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            manager = self.query_cudagraph_manager
            staging = self.staging
            assert manager is not None
            assert staging is not None
            # One graph covers both the collapse of the draft state and the
            # sequential Markov head; only the relayed state moves in and out.
            staging.copy_in(
                num_query_tokens,
                hidden=result["hidden_states"],
                residual=result["residual"],
                post_mix=result["post_mix"],
                res_mix=result["res_mix"],
                pre_mix=result["pre_mix"],
            )
            self._note_graph_replay()
            with phase_profile_stage(
                "draft.markov_total", input_batch.input_ids, level="stage"
            ):
                manager.run_fullgraph(batch_desc)
        else:
            with phase_profile_stage(
                "draft.head.finish", input_batch.input_ids, level="stage"
            ):
                head_hidden = self.model.finish_pipeline_query(
                    result["hidden_states"],
                    result["residual"],
                    result["post_mix"],
                    result["res_mix"],
                    result["pre_mix"],
                )
            with phase_profile_stage("draft.markov_total", head_hidden, level="stage"):
                self._sample_sequential(num_reqs, head_hidden)
        return self.draft_tokens[:num_reqs]

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if self.draft_logits is None:
            return self.model.map_draft_to_target(logits.argmax(dim=-1))

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        return gumbel_sample(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return

        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        with phase_profile_stage("draft.hidden_gather", head_hidden, flow="dspark"):
            sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        with phase_profile_stage("draft.base_logits", sample_hidden, flow="dspark"):
            base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
        sample_hidden_by_step = sample_hidden.view(num_reqs, n_spec, -1)
        self._capture_reference_tensor("draft_head_hidden", sample_hidden_by_step)
        self._capture_reference_tensor("draft_base_logits", base_logits)
        if self._reference_dump_active:
            assert self._reference_dump is not None
            self._reference_dump["draft_previous_ids"] = []
            self._reference_dump["draft_markov_bias"] = []
            self._reference_dump["draft_final_logits"] = []

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.enable_adaptive_verification:
                confidence_markov_embeds.append(markov_embed)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.enable_adaptive_verification:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Apply the sequential Markov head only to top-k base-logit candidates.

        Candidate selection is done once for all draft positions. At each
        sequential step, the selected logits are corrected in place and every
        other entry is set to ``-inf``. The normal dense sampling and rejection
        paths then consume that truncated distribution unchanged.
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # Reuse the dense backbone output as the normal sampler's input. Fill
        # once for all positions, then scatter only the corrected candidates
        # during the sequential loop.
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.enable_adaptive_verification:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.enable_adaptive_verification:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        with phase_profile_stage(
            "draft.backbone", self.input_buffers.input_ids, flow="dspark", level="stage"
        ):
            head_hidden = self._run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        with phase_profile_stage(
            "draft.markov_total", head_hidden, flow="dspark", level="stage"
        ):
            self._sample_sequential(num_reqs, head_hidden)
