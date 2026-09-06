# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import sys
import time
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.b12x_warmup import b12x_warmup
from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    deepseek_v4_mhc_warmup,
)
from vllm.model_executor.warmup.fa4_cutedsl_warmup import (
    fa4_cutedsl_warmup,
)
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
    write_flashinfer_autotune_cache,
)
from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    deepseek_v4_sparse_mla_attention_warmup,
    flashinfer_sparse_mla_decode_autotune_warmup,
)
from vllm.model_executor.warmup.kimi_k3_triton_warmup import (
    kimi_k3_triton_warmup,
)
from vllm.model_executor.warmup.qwen_triton_warmup import qwen_triton_warmup
from vllm.model_executor.warmup.sparse_mla_triton_warmup import (
    sparse_mla_triton_warmup,
)
from vllm.model_executor.warmup.spec_decode_rejection_warmup import (
    spec_decode_rejection_warmup,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_LL_BF16_WARMUP_M_RANGE = range(1, 17)


def _warmup_v2_spec_decode_triton_kernels(runner: "GPUModelRunner") -> None:
    spec_config = runner.speculative_config
    if spec_config is None:
        return

    rejection_sampler = getattr(runner, "rejection_sampler", None)
    vocab_size = getattr(runner, "vocab_size", None)
    if vocab_size is None:
        input_batch = getattr(runner, "input_batch", None)
        vocab_size = getattr(input_batch, "vocab_size", None)
    if rejection_sampler is not None and vocab_size is not None:
        logger.info_once("Warming up V2 speculative decoding Triton kernels.")
        warmup_rejection = getattr(rejection_sampler, "warmup", None)
        if warmup_rejection is not None:
            warmup_rejection(
                vocab_size=vocab_size,
                dtype=runner.model_config.dtype,
                max_num_reqs=runner.max_num_reqs,
            )

    speculator = getattr(runner, "speculator", None)
    if speculator is None:
        speculator = getattr(runner, "drafter", None)
    warmup_prepare_inputs = getattr(speculator, "warmup_prepare_inputs", None)
    use_dflash = getattr(spec_config, "use_dflash", lambda: False)
    use_dspark = getattr(spec_config, "use_dspark", lambda: False)
    if (use_dflash() or use_dspark()) and warmup_prepare_inputs is not None:
        warmup_prepare_inputs()

    _warmup_v2_spec_decode_postprocess_kernels(runner)


def _warmup_v2_spec_decode_postprocess_kernels(runner: "GPUModelRunner") -> None:
    req_states = getattr(runner, "req_states", None)
    model_state = getattr(runner, "model_state", None)
    device = getattr(runner, "device", None)
    spec_config = getattr(runner, "speculative_config", None)
    if req_states is None or model_state is None or device is None:
        return

    num_spec_tokens = getattr(spec_config, "num_speculative_tokens", 0)
    if num_spec_tokens <= 0:
        return

    from vllm.v1.worker.gpu.input_batch import post_update

    idx_mapping = torch.full((1,), -1, dtype=torch.int64, device=device)
    sampled_tokens = torch.zeros(
        (1, num_spec_tokens + 1), dtype=torch.int64, device=device
    )
    num_sampled = torch.ones(1, dtype=torch.int32, device=device)
    num_rejected = torch.zeros_like(num_sampled)
    query_start_loc = torch.tensor(
        (0, num_spec_tokens + 1), dtype=torch.int32, device=device
    )

    output_bin_counts = None
    sampler = getattr(runner, "sampler", None)
    if getattr(runner, "is_last_pp_rank", False) and sampler is not None:
        output_bin_counts = sampler.penalties_state.output_bin_counts

    post_update_args = (
        idx_mapping,
        req_states.num_computed_tokens.gpu,
        req_states.last_sampled_tokens,
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
    )

    # Non-final PP ranks consume the broadcast result without query offsets;
    # the final rank also executes the normal request-local update with them.
    post_update(
        *post_update_args, None, req_states.all_token_ids.gpu, req_states.total_len.gpu
    )
    if getattr(runner, "is_last_pp_rank", False):
        post_update(
            *post_update_args,
            query_start_loc,
            req_states.all_token_ids.gpu,
            req_states.total_len.gpu,
        )

    num_accepted_tokens = getattr(model_state, "num_accepted_tokens_gpu", None)
    if num_accepted_tokens is not None:
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
            _scatter_num_accepted_kernel,
        )

        _scatter_num_accepted_kernel[(1,)](
            idx_mapping,
            num_sampled,
            num_accepted_tokens,
        )


def _warmup_glm5_vision_kernels(worker: "Worker") -> None:
    """Compile the common GLM-5 image-grid encoder variants before serving."""
    model = worker.get_model()
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "glm5_next":
        return

    visual = getattr(model, "visual", None)
    if visual is None or visual.__class__.__name__ == "StageMissingLayer":
        return

    patch_proj = getattr(getattr(visual, "patch_embed", None), "proj", None)
    weight = getattr(patch_proj, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 5:
        return

    channels_per_patch = int(weight[0].numel())
    device = weight.device
    dtype = weight.dtype
    logger.info_once("Warming up GLM-5 vision kernels for image grids 28x28 and 38x38.")
    with torch.inference_mode():
        for grid_size in (28, 38):
            pixel_values = torch.zeros(
                (grid_size * grid_size, channels_per_patch),
                dtype=dtype,
                device=device,
            )
            image_grid_thw = torch.tensor(
                [[1, grid_size, grid_size]], dtype=torch.int32, device=device
            )
            model.embed_multimodal(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        torch.accelerator.synchronize()
    del pixel_values, image_grid_thw
    torch.accelerator.empty_cache()


def _warmup_spec_decode_triton_kernels(runner: "GPUModelRunner") -> None:
    """Compile speculative kernels whose runtime variants dummy runs miss."""
    _warmup_v2_spec_decode_triton_kernels(runner)

    spec_config = runner.speculative_config
    rejection_sampler = getattr(runner, "rejection_sampler", None)
    input_batch = getattr(runner, "input_batch", None)
    if spec_config is None or rejection_sampler is None or input_batch is None:
        return

    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.sample.rejection_sampler import rejection_greedy_sample_kernel
    from vllm.v1.spec_decode.utils import (
        PADDING_SLOT_ID,
        eagle_prepare_inputs_padded_kernel,
        eagle_prepare_next_token_padded_kernel,
        eagle_step_slot_mapping_metadata_kernel,
        next_power_of_2,
    )

    batch_size = runner.max_num_reqs
    num_spec_tokens = spec_config.num_speculative_tokens
    num_sampled_tokens = num_spec_tokens + 1
    device = runner.device
    logger.info_once("Warming up speculative decoding Triton kernels.")

    cu_num_draft_tokens = (
        torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
        * num_spec_tokens
    )
    num_draft_tokens = batch_size * num_spec_tokens
    output_token_ids = torch.full(
        (batch_size, num_sampled_tokens),
        -1,
        dtype=torch.int32,
        device=device,
    )
    draft_token_ids = torch.zeros(num_draft_tokens, dtype=torch.int32, device=device)
    target_argmax = torch.zeros(num_draft_tokens, dtype=torch.int64, device=device)
    bonus_token_ids = torch.zeros(batch_size, dtype=torch.int32, device=device)
    synthetic_conditional_rates = rejection_sampler.synthetic_conditional_rates
    synthetic_mode = synthetic_conditional_rates is not None
    uniform_probs = torch.zeros_like(draft_token_ids, dtype=torch.float32)

    # Rejection sampling switches to the Triton top-k/top-p path at eight rows.
    if num_draft_tokens >= 8:
        sampling_rows = 8
        vocab_size = input_batch.vocab_size
        sampling_logits = torch.zeros(
            (sampling_rows, vocab_size), dtype=torch.float32, device=device
        )
        top_k = torch.full(
            (sampling_rows,),
            max(vocab_size - 1, 1),
            dtype=torch.int32,
            device=device,
        )
        top_p = torch.full((sampling_rows,), 0.9, dtype=torch.float32, device=device)
        for k, p in ((top_k, None), (None, top_p), (top_k, top_p)):
            sampling_logits.zero_()
            apply_top_k_top_p(sampling_logits, k, p)

    rejection_greedy_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        None,
        num_spec_tokens,
        uniform_probs if synthetic_mode else None,
        synthetic_conditional_rates,
        SYNTHETIC_MODE=synthetic_mode,
    )

    drafter = getattr(runner, "drafter", None)
    if (
        spec_config.disable_padded_drafter_batch
        or drafter is None
        or not hasattr(drafter, "prepare_next_token_ids_padded")
        or not hasattr(drafter, "prepare_inputs_padded")
    ):
        return

    warmup_batch_sizes = sorted({1, batch_size})
    for warmup_batch_size in warmup_batch_sizes:
        discard_request_mask = torch.zeros(
            warmup_batch_size, dtype=torch.bool, device=device
        )
        backup_token_ids = torch.zeros(
            warmup_batch_size, dtype=torch.int32, device=device
        )
        next_token_ids = torch.empty_like(backup_token_ids)
        valid_sampled_tokens_count = torch.empty_like(backup_token_ids)
        for sampled_width in (1, num_sampled_tokens):
            sampled_token_ids = torch.zeros(
                (warmup_batch_size, sampled_width),
                dtype=torch.int32,
                device=device,
            )
            eagle_prepare_next_token_padded_kernel[(warmup_batch_size,)](
                sampled_token_ids,
                discard_request_mask,
                backup_token_ids,
                next_token_ids,
                valid_sampled_tokens_count,
                input_batch.vocab_size,
                sampled_width,
                warmup_batch_size,
                sampled_token_ids.stride(0),
                BLOCK_SIZE_TOKENS=next_power_of_2(sampled_width),
            )

        query_start_loc = (
            torch.arange(warmup_batch_size + 1, dtype=torch.int32, device=device)
            * num_sampled_tokens
        )
        token_indices_to_sample = torch.empty_like(backup_token_ids)
        num_rejected_tokens = torch.empty_like(backup_token_ids)
        eagle_prepare_inputs_padded_kernel[(warmup_batch_size,)](
            cu_num_draft_tokens[:warmup_batch_size],
            valid_sampled_tokens_count,
            query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens,
            warmup_batch_size,
        )

    block_size = getattr(drafter, "block_size", -1)
    kv_cache_gid = getattr(drafter, "kv_cache_gid", -1)
    block_tables = input_batch.block_table.block_tables
    if block_size <= 0 or not 0 <= kv_cache_gid < len(block_tables):
        return

    for warmup_batch_size in warmup_batch_sizes:
        block_table = block_tables[kv_cache_gid].block_table.gpu[:warmup_batch_size]
        positions = torch.zeros(warmup_batch_size, dtype=torch.int64, device=device)
        seq_lens = torch.zeros(warmup_batch_size, dtype=torch.int32, device=device)
        clamped_positions = torch.empty_like(positions)
        slot_mapping = torch.empty(warmup_batch_size, dtype=torch.int64, device=device)
        eagle_step_slot_mapping_metadata_kernel[(warmup_batch_size,)](
            positions,
            block_table,
            block_table.stride(0),
            seq_lens,
            clamped_positions,
            slot_mapping,
            block_size=block_size,
            max_model_len=drafter.max_model_len,
            n_blocks_per_req=block_table.shape[1],
            PAD_ID=PADDING_SLOT_ID,
            batch_size=warmup_batch_size,
        )


def _ll_bf16_router_shapes_from_model(
    model: torch.nn.Module,
) -> tuple[tuple[int, int], ...]:
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    shapes: set[tuple[int, int]] = set()
    for module in model.modules():
        if not isinstance(module, GateLinear):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if weight.dim() != 2 or weight.dtype != torch.bfloat16:
            continue
        n, k = weight.shape
        if k % 8 == 0:
            shapes.add((int(k), int(n)))
    return tuple(sorted(shapes))


def _warmup_ll_bf16_router_gemm(model: torch.nn.Module) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        is_available as is_ll_bf16_gemm_available,
    )
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        ll_bf16_gemm_kernel,
    )

    if not is_ll_bf16_gemm_available():
        return

    shapes = _ll_bf16_router_shapes_from_model(model)
    if not shapes:
        logger.debug_once(
            "Skipping ll_bf16 router GEMM warmup: no bf16 GateLinear shapes found."
        )
        return

    logger.info_once("Warming up ll_bf16 router GEMM kernels for shapes: %s.", shapes)
    ll_bf16_gemm_kernel.warmup(
        shapes=shapes,
        m_values=_LL_BF16_WARMUP_M_RANGE,
    )


def _warmup_kimi_k3_gemm_rs_ar() -> None:
    # Kimi-K3 model construction imports this module only when GEMM-RS/AR is
    # enabled and initializes its singleton before kernel_warmup runs. Avoid
    # importing it here so other models do not compile the RS/AR variants.
    module = sys.modules.get("vllm.models.kimi_k3.nvidia.ops.cute_dsl.gemm_rs_ar")
    if module is None:
        return
    compiled = module.warmup_gemm_rs_ar()
    if compiled:
        logger.info_once("Warmed up %d Kimi-K3 GEMM-RS/AR variants.", compiled)


def kernel_warmup(worker: "Worker", *, process_local_only: bool = False):
    from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
        minimax_m3_msa_warmup,
    )

    if not worker.use_v2_model_runner:
        # The KV-block zeroing kernel is driven by the scheduler's
        # `new_block_ids_to_zero`, so no dummy run ever reaches it.
        zeroer = getattr(worker.model_runner, "_kv_block_zeroer", None)
        if zeroer is not None:
            zeroer.warmup(worker.model_runner.kv_cache_config.num_blocks)

    if worker.vllm_config.kernel_config.enable_jit_warmup:
        logger.info("JIT kernel warmup starting.")
        jit_warmup_start = time.perf_counter()
        try:
            worker.model_runner.jit_warmup_registry.warmup()
        except Exception:
            logger.exception(
                "JIT kernel warmup failed after %.2fs.",
                time.perf_counter() - jit_warmup_start,
            )
            raise
        logger.info(
            "JIT kernel warmup finished in %.2fs.",
            time.perf_counter() - jit_warmup_start,
        )

    qwen_triton_warmup(worker.model_runner, worker.vllm_config.model_config)

    compilation_config = worker.vllm_config.compilation_config
    cudagraph_capture_sizes = list(compilation_config.cudagraph_capture_sizes or [])

    # DSv4 mHC TileLang kernels (hc_pre/hc_post/hc_head_op) run every decoder
    # layer per token; warm them across token sizes first so the first real
    # request doesn't pay JIT cost. No-op for non-DSv4 models (gated inside).
    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
    )
    _warmup_glm5_vision_kernels(worker)

    # Run next so input-prep kernels JIT against pristine runner state.
    if worker.vllm_config.kernel_config.enable_jit_warmup:
        _warmup_spec_decode_triton_kernels(worker.model_runner)
        kimi_k3_triton_warmup(worker)
        fa4_cutedsl_warmup(worker)
        sparse_mla_triton_warmup(worker)
        spec_decode_rejection_warmup(worker)

    if current_platform.has_device_capability(90):
        _warmup_ll_bf16_router_gemm(worker.get_model())

    _warmup_kimi_k3_gemm_rs_ar()

    if worker.vllm_config.kernel_config.enable_cutedsl_warmup:
        # TODO(roberto): Remove after registered CuTeDSL warmups are migrated
        # to the shared JIT warmup infrastructure.
        # https://github.com/vllm-project/vllm/pull/47451
        cutedsl_warmup()

    if process_local_only:
        return

    flashinfer_sparse_mla_decode_autotune_warmup(worker)
    deepseek_v4_sparse_mla_attention_warmup(worker)

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        is_deep_gemm_supported() and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    b12x_warmup(worker, cudagraph_capture_sizes)

    minimax_m3_msa_warmup(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info_once("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info_once("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


def _flashinfer_autotune_skip_ops(runner: "GPUModelRunner") -> set[str] | None:
    if envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS is not None:
        return set(envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS) or None

    from vllm.model_executor.kernels.linear import (
        FlashInferCuteDslNvFp4LinearKernel,
    )

    for module in runner.get_model().modules():
        for holder_name in ("quant_method", "scheme"):
            kernel = getattr(getattr(module, holder_name, None), "kernel", None)
            # CuTe-DSL mm_fp4 tuning JIT-compiles every tactic and its
            # fallback is already the heuristic; all mm_fp4 backends share
            # the "fp4_gemm" op name, so skip only when cute-dsl is selected.
            if isinstance(kernel, FlashInferCuteDslNvFp4LinearKernel):
                return {"fp4_gemm"}
    return None


_FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS = 32


def _flashinfer_autotune_token_counts(runner: "GPUModelRunner") -> tuple[int, ...]:
    max_tokens = runner.scheduler_config.max_num_batched_tokens
    linear_backend = runner.vllm_config.kernel_config.linear_backend
    if (
        linear_backend == "flashinfer_cutedsl"
        and max_tokens > _FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS
    ):
        return max_tokens, _FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS
    return (max_tokens,)


def _run_flashinfer_autotune_dummy_runs(runner: "GPUModelRunner") -> None:
    for num_tokens in _flashinfer_autotune_token_counts(runner):
        logger.info("Running FlashInfer autotune with %d tokens.", num_tokens)
        runner._dummy_run(
            num_tokens=num_tokens,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
        )


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Every rank profiles the same tactics. When distributed, per-tactic
    timings are averaged over the world CPU group so all ranks select the
    same tactic.
    """
    from flashinfer.autotuner import AutoTuner, set_autotune_process_group

    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import get_world_group

    world = get_world_group()
    is_leader = world.rank_in_group == 0
    tuner = AutoTuner.get()

    autotune_kwargs: dict = {}
    skip_ops = _flashinfer_autotune_skip_ops(runner)
    if skip_ops:
        logger.info_once(
            "Skipping FlashInfer autotuning for ops %s",
            tuple(sorted(skip_ops)),
        )
        autotune_kwargs["skip_ops"] = skip_ops

    cache_path = resolve_flashinfer_autotune_file(runner)
    if is_leader:
        logger.info_once("Using FlashInfer autotune cache file: %s", cache_path)

    # We skip EPLB here since we don't want to record dummy metrics.
    # Randomize inputs to avoid every token pick the same experts,
    # which lead to some EP ranks receiving no tokens and skipping their
    # MoE kernel entirely, and cause hang due to all-reduce collective
    # during synchronized autotuning.
    # Read cached autotune results and broadcast to all ranks.
    cached_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached_results = f.read()
    cached_results = world.broadcast_object(cached_results, src=0)
    if cached_results is not None:
        write_flashinfer_autotune_cache(cache_path, cached_results)
        world.barrier()
        tuner.load_configs(str(cache_path))

    group = world.cpu_group if world.world_size > 1 else None
    set_autotune_process_group(group)
    try:
        with (
            torch.inference_mode(),
            fi_utils.autotune(tune_mode=True, **autotune_kwargs),
        ):
            _run_flashinfer_autotune_dummy_runs(runner)
    finally:
        set_autotune_process_group(None)

    if world.world_size > 1:
        world.barrier()
    if is_leader:
        tuner.save_configs(str(cache_path))
