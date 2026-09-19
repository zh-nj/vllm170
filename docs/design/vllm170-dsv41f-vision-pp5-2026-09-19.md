# DeepSeek-V4.1-Flash vision on PP5 / SM80 with DSpark

Date: 2026-09-19. Scope: enable image inputs on the port instance, keep the vision
tower on PP0, and verify that the text path does not regress.

Instance: PP5 / TP1 on 5x CMP 170HX (SM80), `mxfp4_int8_sm80` MoE,
`TRITON_MLA_SPARSE_DSV41` attention, FP8 KV, FULL target cuda graphs, DSpark width
5 with `dspark_pipeline_stage_ranks=[1,2,3]`.

## What was already in the tree

Nothing had to be written for the tower itself. `vllm/models/deepseek_v4_1/` already
registers the multimodal entry point, and the architecture name in the checkpoint
resolves to it:

- `nvidia/vl_model.py`: `DeepseekV41ForCausalLM`, `SupportsMultiModal`, with
  `self.owns_vision = get_pp_group().is_first_rank`. Only that rank builds the ViT,
  the aligner and the learned `image_start` / `image_end` / `image_newline`
  sentinels; every other rank gets `StageMissingLayer`.
- `common/mm_preprocess.py`: placeholder expansion, image-span layout, resize
  planning, and the `COMPRESS_PAD_TO` alignment pad.
- The sparse-SWA metadata builder widens the window inside an image span
  in-kernel (`_build_image_visibility` -> `_compute_swa_indices_and_lens_kernel`
  with `HAS_IMAGE=True`), which is the SM80-compatible mechanism: no CUDA-graph
  special case, no backend capability flag.
- SM80 hard-selects `DeepseekV41AmpereMLAAttention` in `nvidia/model.py`, so
  `use_mm_prefix` never goes through the backend selector and no
  `supports_mm_prefix` claim was needed.

## Recipe

```bash
PORT_MM_IMAGES=2 PORT_MAX_BATCHED=1024 ... benchmarks/deepseek_v4_1/run_port_pp5.sh
```

`PORT_MM_IMAGES` (default now 2) sets `--limit-mm-per-prompt`, and the launcher adds
the one override images need when it is non-zero. Two constraints define the
configuration:

1. **A modality forces `--disable-chunked-mm-input`** (bidirectional image spans
   cannot be chunked), which makes the engine require
   `max_tokens_per_mm_item <= max_num_batched_tokens`. For this checkpoint that
   bound is `vision_max_n_token + COMPRESS_PAD_TO - 1 = 1024 + 2 - 1 = 1025`.
2. **Raising the token budget is not free.** With `max_num_batched_tokens`
   1024 -> 1152 the greedy token ids changed on 2 of 5 reference prompts and
   DSpark acceptance fell from 5.69 to 5.06 tokens/step — and, isolated, that
   happens **with and without vision**, so it is the budget, not the images.

So the launcher caps the per-image budget instead:

```bash
--hf-overrides '{"vision_max_n_token":1023}'
```

which puts the item bound at exactly 1024 and lets the text path keep its 1024
geometry. The override must be **flat**: a nested
`{"vision_config": {"max_image_tokens": ...}}` is applied by `_apply_dict_overrides`
*after* `DeepseekV41Config.__init__` has derived `vision_max_n_token` from
`vision_config`, so the nested form silently does nothing (observed: the engine
still refused to start with the 1025 bound).

## Verification

| Check | Result |
|---|---|
| Checkpoint example, two photographs (`inference/examples/example_harmony.json`) | **PASS**: first image 胡萝卜 (肉质直根), second 玉米 (籽粒) |
| Orientation probe, horizontal vs vertical white bar | **2/2** correct, different answers per image |
| Text, 5 reference prompts, greedy token ids vs the text-only baseline | **5/5 identical** |
| Text DSpark acceptance per position | 66 / 66 / 60 / 57 = unchanged |
| Vision weight placement | **PP0 55.10 -> 56.05 GiB (+0.95)**, PP1-PP4 unchanged (57.48 / 56.06 / 56.06 / 54.92) |
| Image prompt with DSpark | 74 drafts, acceptance 48.9%, accepted length 3.446 |

The last row is expected rather than a defect: the speculator feeds the drafter
text-only inputs, and the checkpoint's own reference implementation does the same
(`DSparkBlock.forward`: `# drafts are text: no VL bias`). Speculation therefore
pays less on image prompts (3.4 vs 5.6 tokens/step) but the verified output is
unchanged, because greedy verification is exact.

## Engram is not part of the memory peak

The question of what the Engram tables cost on device, since they are the largest
single allocation in the process:

| Item | Where | Size (per Engram rank) |
|---|---|---|
| n-gram tables (`384M rows x 256`, fp8 + scales) | pinned host RAM | **94.42 GiB** (log: `Engram host table: ... 94.42 GiB per rank`) |
| other Engram weights (`wkv`, q/k mixes) | pinned host RAM | 2.86 GiB per rank |
| UVA views of both | device | 0 (pointers only; `_storage()` returns `get_accelerator_view_from_cpu_tensor` when `cpu_offload`) |
| `Engram.staged_rows` gather output | device | 12.58 MB (1024 x 24 x 256 bf16) |
| slot cache | device | ~2 MB (KV tokens x int32) |
| `lookup` output temporary | device | 12.58 MB per call, freed immediately |

So the whole device-side Engram footprint is ~27 MB per Engram rank, and the
"Copying Engram in 64 MiB chunks" loader is a host-to-host copy
(`param.data[start:start+count].copy_(shard[...])` with a pinned destination), not
a staging buffer. The 189 GiB is host RAM holding the weights themselves, which
cannot move to a 63 GiB card; `cpu_cache_gb=0` and `ssd_offload_path=None` mean
there is no second copy.

## The startup OOM was ordering, not size

Startup used to die in the mixed prefill+decode warmup with
`ncclUnhandledCudaError ... Cuda failure 2 'out of memory'` on the largest rank,
and the workaround was to trim NCCL's buffers (channels, then `NCCL_BUFFSIZE`).
The traceback names the real cause: the warmup issues the **first collective** on
the PP token-broadcast subgroup, and NCCL allocates a communicator's device
buffers on first use, not at creation. By then the card holds 57.5 GiB of weights,
the KV cache and the warmup workspaces, leaving a few hundred MiB.

Fix, in `vllm/v1/worker/gpu/pp_utils.py` and `gpu_worker.py`:

1. `PPHandler.__init__` warms the freshly created `pp_broadcast` sibling group and
   the PP stage group with a one-element broadcast
   (`warm_collective_group`). This happens during model-runner construction —
   after weights, before the KV cache — so the same buffers are allocated while
   gigabytes are still free.
2. `compile_or_warm_up_model` calls `gc.collect()` +
   `torch.accelerator.empty_cache()` before `warmup_kernels`, handing
   cached-but-unused blocks back to the driver, since NCCL allocates with raw
   `cudaMalloc` and cannot see the torch cache.

Result: the instance starts on the first attempt with **default** NCCL buffers
(`NCCL_BUFFSIZE=4194304`, channels auto) even with vision enabled — a
configuration that previously failed every attempt. The launcher still defaults to
`NCCL_BUFFSIZE=1 MiB` / `NCCL_MAX_NCHANNELS=8`, which is now a performance
setting rather than an OOM workaround: measured DSpark-on, 84.5 ms/step with the
trim against 88.4-90.1 ms/step without it in the same hour. That difference is
within this box's drift band but consistent in sign, and all the DSpark ledgers in
`vllm170-dsv41f-dspark-flow-ledger-2026-09-18.md` were measured with the trim.
