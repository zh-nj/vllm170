# `max_num_batched_tokens`: text output, prefill speed, and images

Ledger for the DSV4.1-Flash port on 5x CMP 170HX (SM80), PP5/TP1, DSpark
`num_speculative_tokens=5`. Companion to
`vllm170-dsv41f-prefix-caching-2026-09-19.md`, which pins the three
shape-selected kernels this page builds on.

## Verdict

`--max-num-batched-tokens` is a *numeric* parameter of this deployment, not only
a performance knob, and the reason is a single classification:

* the scheduler caps a prefill chunk at `max_num_batched_tokens -
  max_num_new_slots_for_drafting` rows, so chunk 1 is `budget - 4` with DSpark
  (1020 at the default 1024, 1148 at 1152), and the tail of a prompt of length
  `L` is whatever is left: `L mod (budget - 4)`;
* a prefill chunk no wider than `decode_threshold` is **classified as a decode**
  and scored by the decode kernels, which do not round like the prefill kernels.
  So which prompts have such a tail -- and therefore which kernel scores their
  last tokens -- follows the budget;
* two backends split at different widths (the sparse-SWA attention at
  `1 + 2 * num_speculative_tokens = 11`, the sparse indexer at
  `num_speculative_tokens + 1 = 6`), so before the fix they disagreed about the
  rows in between, and both sent the rows below 6 to the decode side.

`dsv41_strict_prefill_tail: true` puts both splits on the prefill side for a
chunked-prefill tail. Verified below by pairwise tail invariance and by
bit-identical hidden states, after which the budget no longer moves the model.

The historical observation in `benchmarks/deepseek_v4_1/run_port_pp5.sh` (1152
changed the greedy token ids on 2 of 5 reference prompts, DSpark acceptance
5.69 -> 5.06) predates the three shape-selected kernels being pinned; with
`marlin_full_k`, `dsv41_strict_bf16_gemm` and now `dsv41_strict_prefill_tail` on,
no budget from 1024 to 4096 changes a greedy token, a reference prompt or an
acceptance counter. The budget is a performance knob again, and **2048** is the
best measured setting.

## How the budget becomes a chunk width

`vllm/v1/core/sched/scheduler.py:536,628` takes `input_budget =
max_num_batched_tokens` and admits `min(token_budget, input_budget -
draft_slots)`; `draft_slots = speculative_config.max_num_new_slots_for_drafting`
(`:535`), which is `num_draft_tokens - 1` for a parallel-drafting config that
reuses the existing query (`vllm/config/speculative.py:1981-1988`). DSpark sets
`parallel_drafting = True` (`speculative.py:1518`), so with
`num_speculative_tokens = 5` the reservation is 4 and chunk 1 is `budget - 4`.
The same `4` shows up in `sparse_swa.py`'s `decode_threshold` as
`1 + 2 * 5 = 11`: `_init_reorder_batch_threshold` (`backend.py:627-655`) raises
the runner's reorder threshold to `1 + (2 if parallel_drafting else 1) *
num_speculative_tokens`, so the batch is ordered decode -> short_extend ->
long_extend -> prefill at exactly the width the attention splits at.

DSpark is on by default in this port (`PORT_DSPARK_TOKENS=5`) and was on for
every measurement on this page.

## Finding 1 -- a short prefill tail is scored by the decode kernels

`split_decodes_and_prefills` (`vllm/v1/attention/backends/utils.py`) classifies a
request by query length unless told otherwise:

```text
treat_short_extends_as_decodes: If True (default), short extends
    (query_len <= threshold but still prefilling) are counted as decodes.
```

A *short extend* is a chunked-prefill tail no wider than the threshold. Counting
it as a decode routes those rows to `rocm_sparse_attn_decode` instead of
`rocm_sparse_attn_prefill` (`vllm/models/deepseek_v4_1/amd/rocm.py:760,820`), two
kernels that do not agree to the last bit.

Measured at a fixed budget of 1024 (chunk 1 = 1020), sweeping the tail by
lengthening the prompt one token at a time -- the tail is the only thing that
changes:

| tail rows | 1 | 2 | 3 | 4 | 5 | 6..11 | >= 12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logits vs a 80-row tail | = | X | X | X | X | X | = |

Every tail up to 11 rows differed from a 80-row tail and every tail from 12 rows
up matched it, which is `decode_threshold = 11` exactly. Positions that moved
were the tail's rows *after the first* -- 3 positions for a 4-row tail, 6 for a
7-row tail -- the first row of a decode-classified chunk matches the prefill
kernels' value.

Rather than reconstruct which of the two kernels was "right", the fix removes the
choice: a chunked-prefill tail is a prefill, whatever its width.

## Fix -- `dsv41_strict_prefill_tail`

`additional_config: {"dsv41_strict_prefill_tail": true}` (JSON boolean, validated;
default off, so the budget keeps its previous numerics). It is read by
`strict_prefill_tail` (`vllm/v1/attention/backends/utils.py`) at both split
sites:

* `DeepseekSparseSWAMetadataBuilder` (`vllm/v1/attention/backends/mla/sparse_swa.py:458`),
  the attention's split at `decode_threshold = 11`;
* `DeepseekV32IndexerMetadataBuilder` (`vllm/v1/attention/backends/mla/indexer.py:728`),
  the indexer's split at `decode_threshold = next_n = 6`.

Both are needed. With only the attention side changed, the indexer still called
tails of 1..5 decodes and the two disagreed: the *support set* the indexer
selected stopped matching what the attention expected, and a 4-row tail diverged
from a 12-row tail from layer 2 onward, growing to 13.0 in the last layer's
residual. With both sides changed the disagreement is gone.

`is_prefilling` (`CommonAttentionMetadata`, "True if request is still in prefill
phase") is what tells a short extend from a real decode; when the runner does not
supply it the old behavior is kept, and the reorder guarantees the batch order the
split assumes.

Two further `split_decodes_and_prefills` call sites pass
`treat_short_extends_as_decodes=not self.use_pcp`
(`model_executor/layers/attention/sparse_mla_attention.py:276`,
`mla_attention.py:2446`). They belong to other MLA implementations, not to the
sparse-SWA path this model runs, and were left alone: closing the tail band with
only the two sites above is the evidence that they are off this path.

### Verification

Pairwise, within one boot, positions at and after the second row of the tail, for
tails of 2..18 rows: every pair is identical with both splits changed, where the
attention-only change left the 2..5 band moved:

```text
                2    4    6    8   10   12   14   16   18
  attention only  .    0    1    1    1    1    1    1    1
                 .    .    3    3    3    3    3    3    3
                 .    .    .    0    0    0    0    0    0
                 .    .    .    .    0    0    0    0    0
  both           all zero
```

Hidden states, dumped per layer and stage by the row probe
(`/tmp/dsv41_rowdump_*`) at position 1150 on all five ranks:

| comparison | stages | differ |
| --- | --- | --- |
| 4-row tail vs 12-row tail, both splits changed | 148 | 3, all addressing (`support_indices`, `context_rows`, `positions` -- a constant 8-row shift of the gather region) |
| 4-row tail vs 900-row tail, attention side only (before) | 148 | 140, first at layer 2 `attn_out` (7.8e-03), last 1.3e+01 |
| 12-row tail vs 900-row tail, attention side only | 148 | 3, all addressing |

So a tail's width no longer reaches the model at all.

## Finding 2 -- geometry keyed on 1024 is inert in practice

Raising the budget to 1152 also moves chunk 1 from 1020 to 1148 rows, crossing
several thresholds at exactly 1024. All are either pinned or measured inert:

* **MHC pre-norm** picks `hc_prenorm_gemm_block_m_tilelang` at `x.shape[0] >= 1024`
  (`vllm/model_executor/kernels/mhc/tilelang.py:53`) -- guarded by
  `batch_invariant_mode_enabled()`, which `dsv41_strict_bf16_gemm` switches on.
* **Sparse indexer query-row cap** `VLLM_SPARSE_INDEXER_MAX_QUERY_ROWS = 1024`
  (`vllm/envs.py:1171`): a 1148-row chunk is served as 1024 + 124 with
  `skip_kv_gather=True` on the second. Measured inert: for prompts of 1100, 1148
  and 1533 tokens the first 1020 positions are identical across budgets, i.e.
  across one indexer chunk and two.
* **Multi-stream GEMM** `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD = 1024`
  (`vllm/models/deepseek_v4_1/attention.py:867`): a 1020-row chunk overlaps the
  fused qkv projection with the compressor and indexer projections, a 1148-row
  chunk does not. Measured inert in the same comparison.
* **MoE tile selector** flips at `M = 1025` (`average = ceil(M * topk /
  num_experts)` with 384 experts and topk 6). Arithmetic-identical: the grouped
  GEMM fixes ThreadblockK at 128, so only the output tiling and the padding
  change, and padded rows carry weight 0.

Marlin's dependence is continuous in `M` through its split points rather than
thresholded, and the split points are what moves: `prob_m_split` is 960 at
`M = 1020` and 1088 at `M = 1148` (`csrc/.../marlin/marlin.cu:465-469`), which
makes the kernel's own `parallel = prob_m / m_block_size` 15 against 17 and the
stream-K decomposition `iters` 40 against 56 at `prob_n = 1792` -- a different
K-stripe boundary per row. Nothing else in the launch discriminates these two
sizes: `thread_m_blocks` is 4 on every launch of both, so the thread-config table
and the `red_off` cross-warp reduce are identical, and `prob_m` is never read by
`get_scales_cache_size` / `get_kernel_cache_size`, so the auto exec-config choice
is `M`-independent. `marlin_full_k` is therefore the sufficient pin and the only
one needed: it makes `global_mn_tiles` equal `gridDim.x`, so `iters = k_tiles`
and `parallel` stops mattering. cuBLAS's kernel choice is likewise continuous in
`M`, with no 1024-row boundary, and is pinned by `dsv41_strict_bf16_gemm`.

The most sensitive of those call sites is the MoE router gate:
`vllm/model_executor/layers/fused_moe/router/gate_linear.py:424` calls
`torch.mm(x, weight.T, out_dtype=torch.float32)` -- the `aten::mm.dtype`
overload -- and on SM80 that tier is the router's exclusive path, so its cuBLAS
kernel regime steps every 64 rows of `M` (1020 falls in 961..1024, 1148 in
1089..1152). A router logit delta there is not just another hidden-state
rounding: `router/dsv4_topk.py` turns it into a discrete expert-set change that
the MoE then computes with different weights. `dsv41_strict_bf16_gemm` pins the
overload, which is why the same position survives `M` 1020 -> 1148 -> 2044 here;
without that lever the port would need `VLLM_DSV41_ROUTER_GEMM_ALIGN=split`,
which its PP5 launchers do not set.

## Finding 3 -- `prompt_logprobs` is not a cross-shape comparator

The gates compare per-position logits, which needs `prompt_logprobs`; that path
computes logits for *every* row of the chunk (M = chunk width), where generation
computes them for the sampled row only. Its values are therefore shape- and
state-dependent in a way the model is not:

* at position 1500 with a 892-row and a 900-row tail, all 148 dumped hidden-state
  stages are bit-identical while the prompt-logprobs values differ at 891
  positions by up to 2.4e-02 -- about one bf16 ULP at those logit magnitudes;
* the same position, same request, differs between server boots (896 positions)
  although the dumped hidden states are again bit-identical, and the first
  request after a boot differs from later ones on the same boot;
* greedy token ids, which read the sampled row's logits only, never moved.

So a `prompt_logprobs` difference at large M is not evidence about generation,
and per-position fingerprint comparisons are only trustworthy within one boot and
for nearby shapes. This is why the tail sweep above compares tails pairwise
rather than against one long reference: the 900-row reference is the part that
wobbles. Greedy ids and row dumps are the instruments that hold.

## Best value

Prefill, cold prompt, five repeats, median (`benchmarks/deepseek_v4_1/prefill_gate.py`
with `--max-tokens 1 --idle-s 3`; without the idle the median comes out bimodal
because the previous request is still decoding):

| budget | chunk 1 | 32K TTFT | 32K tok/s | 8K TTFT | 4K TTFT | marginal rate | idle mem, busiest rank | free |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1020 | 7.389 s | 4435 | 2.463 s | 1.653 s | 4989 tok/s | 61120 MiB | 3.9 GiB |
| 1152 | 1148 | 7.041 s | 4654 | 2.385 s | 1.615 s | 5278 tok/s | -- | -- |
| **2048** | **2044** | **6.67-6.81 s** | **4810-4916** | 2.616 s | 1.949 s | 5857 tok/s | 61998-63222 MiB | 2.4-3.0 GiB |
| 4096 | 4092 | 7.30-7.32 s | 4479-4490 | 3.409 s | 2.721 s | 6292 tok/s | 64214 MiB | 0.79 GiB |

TTFT is `a + len / rate` with the fitted `a` growing with the budget (0.82 s at
1024, 0.83 at 1152, 1.22 at 2048, 2.11 at 4096), which is why the wider chunks
stop paying for themselves: 4096 has the best marginal rate and the worst TTFT,
and a 4K prompt is slower there (2.72 s) than at the 1024 default (1.65 s).

Quality, measured at every one of those budgets, changed nowhere:

* sixteen prompt lengths (900 .. 2048, covering tails of 1..1028 rows) produce
  **identical 64-token greedy continuations** across 1024 / 1152 / 2048 / 4096;
* the launcher's own five reference prompts (`prefix_cache_gate.py
  --seed-base 20260919`) produce **identical 96-token continuations** at 1024 and
  at 4096 -- the same comparison that read "2 of 5 changed" before. Per-case
  DSpark accepted length is identical too, except `shared-prefix`, which improves
  (3.333 -> 3.448);
* DSpark acceptance on the 32K gate is bit-identical at 1024, 2048 and 4096 for
  all three seeds tried: `[11,11,11,10,9]`, `[11,11,11,11,10]`, `[12,12,11,10,10]`
  accepted per draft position, i.e. 94.55% / 98.18% / 90.00% and accepted length
  5.727 / 5.909 / 5.500.

Under a concurrent decode the budget is a cost, not a gain, so the pick depends
on which half matters (`mixed_gate.py`: a 256-token request streaming 48 tokens,
then a cold 32K prefill launched 0.6 s later; three repeats each, spread under
5%):

| budget | decode tokens served during the ~9 s prefill | median gap | worst gap | quiet ITL | solo 32K TTFT |
| --- | --- | --- | --- | --- | --- |
| 1024 | 6 | 1486 ms | 1615 ms | 86 ms | 7.35 s |
| 2048 | 5 | 1866 ms | 2836 ms | 83 ms | 6.7-6.8 s |

Both budgets starve the decode -- it is served about once per prefill chunk --
because a long prefill takes the whole per-request budget and a DSpark decode
step needs `1 + num_speculative_tokens = 6` tokens of it, so the two rarely share
a step; the scheduler's prefill throttling lets a few decode-only steps through.
The wider budget widens the gap (worst case 1.6 s -> 2.8 s) while buying 8% on
TTFT.

So **2048** when prefill throughput and image room matter more than a concurrent
decode's latency -- the fastest measured point (32K TTFT 6.67-6.81 s across
three boots, 4810-4916 tok/s), images up to 2042 tokens instead of 1018, and
2.4-3.0 GiB still free on the busiest rank (PP2, which carries a DSpark draft
stage) -- and **1024 or 1152** when a decode running beside a long prefill has to
stay under about 1.5 s.
4096 was expected to be the better answer for prefill throughput and is not -- it
is slower end to end and leaves 0.79 GiB. 1025 is the smallest budget that
restores the checkpoint's own 1023-token image budget. 1152 is the value the
launcher's comment named; it is dominated by 2048 on both speed and image room.

End to end, with the derived cap: a 4096x4096 image now yields 2019 prompt
tokens (about 2008 of them image tokens) where the port could previously reach
1018-plus-pad, an 8000x8000 image is clamped to the same grid, and both the
vision gate and the reference five prompts still pass.

One ceiling to keep in mind before raising the budget further: the indexer's
per-row top-k switches from insertion to radix selection at `numRows > 12288`
(`csr/.../sampler.cu` `top_k_per_row_prefill`), so a chunk wider than that would
change the selection algorithm mid-batch. 2048 and 4096 are far below it; a
megatoken budget would not be.

The port launcher now derives the image cap from the budget *and* from the
drafting reservation: `PORT_VISION_MAX_N_TOKEN` defaults to
`PORT_MAX_BATCHED - 1 - (PORT_DSPARK_TOKENS - 1)`, because an image span is
scheduled atomically and the chunk a request gets is the budget minus the slots
DSpark reserves -- 1019 at the default 1024, 2043 at 2048. The flat 1023 this
launcher shipped left a four-token window (1021..1024) in which an image item
would not fit any chunk, and `scheduler.py:1749-1762` rolls such an item back and
defers it on every step, which stalls the request rather than failing it. The cap
itself is inert for the text path: at a fixed budget of 2048, raising it from
1023 to 2047 left every per-position logit of all sixteen prompts
**bit-identical**, and the vision gate passes with the wider grid.

The constraint that forces the budget up at all is multimodal:
`encoder_cache_manager.py:324` requires `max_tokens_per_mm_item <=
max_num_batched_tokens` whenever `disable_chunked_mm_input` is set, which
`vllm/platforms/cuda.py:337` forces for a prefix-LM model, and an image item is
bounded by `vision_max_n_token + COMPRESS_PAD_TO - 1`
(`vllm/models/deepseek_v4_1/common/mm_preprocess.py:344`, `COMPRESS_PAD_TO = 2`).
`vision_config.max_image_tokens = 1024` in the checkpoint's own `config.json`
(`vllm/transformers_utils/configs/deepseek_v41.py:60` reads it as
`vision_max_n_token`), of which `safe_resize` reserves `COMPRESS_PAD_TO - 1` for
the compressor pad, so the checkpoint's own per-image budget is 1023 tokens and
the budget it needs is 1025. The port launcher's `vision_max_n_token = 1023`
override brings the item bound down to exactly 1024 -- images lose one token to
keep the budget at its default.

## Reproducing

```bash
# same-config determinism (must be bit-identical), then cross-config comparison
python benchmarks/deepseek_v4_1/budget_gate.py --lengths 900 1020 1021 1024 \
    1030 1031 1032 1100 1148 1152 1155 1160 1301 1533 2040 2048 \
    --max-tokens 64 --out /tmp/budget.json
python benchmarks/deepseek_v4_1/budget_gate.py --within /tmp/budget.json
python benchmarks/deepseek_v4_1/budget_gate.py --compare A.json B.json

# tail band: tails of 1..18 rows against each other
python benchmarks/deepseek_v4_1/budget_gate.py --lengths $(seq 1149 1166) 2048 \
    --max-tokens 4 --out /tmp/tails.json
```

Restart with the lever:

```bash
bash benchmarks/deepseek_v4_1/stop_pp5.sh
bash benchmarks/deepseek_v4_1/start_pp5_patient.sh /tmp/dsv41_server.log \
    PORT_MAX_BATCHED=1152 PORT_PREFIX_CACHING=1 PORT_KV_BYTES=700000000 \
    PORT_MAX_MODEL_LEN=262144 'PORT_CUDAGRAPH_SIZES=[1,2,4,6,8,12,16,24,32,40]' \
    'PORT_EXTRA_ARGS=--additional-config {"marlin_full_k":true,"dsv41_strict_bf16_gemm":true,"dsv41_strict_prefill_tail":true}'
```

Tests:

```bash
.venv/bin/python -m pytest tests/v1/attention/test_attention_splitting.py \
    tests/v1/attention/test_batch_reordering.py -q
```

## Files

* `vllm/v1/attention/backends/utils.py` -- `strict_prefill_tail` and its
  `split_decodes_and_prefills` contract.
* `vllm/v1/attention/backends/mla/sparse_swa.py` -- attention-side split and the
  `treat_short_extends_as_decodes` class attribute.
* `vllm/v1/attention/backends/mla/indexer.py` -- indexer-side split.
* `vllm/models/deepseek_v4_1/nvidia/model.py` -- the row dump no longer skips
  batches below 64 rows, which is what hid the short-tail case.
* `benchmarks/deepseek_v4_1/budget_gate.py` -- per-position logits and greedy
  ids, `--within` (same-config batch invariance) and `--compare` (cross-config).
* `benchmarks/deepseek_v4_1/prefill_gate.py` -- cold-prompt TTFT and prefill
  tokens/s, with a per-run salt nonce, an idle before each repeat and the
  prefix-cache hit counter read around every request.
* `benchmarks/deepseek_v4_1/mixed_gate.py` -- what a cold 32K prefill does to a
  decode running beside it: decode tokens served inside the prefill window, and
  the gaps between them.
* `benchmarks/deepseek_v4_1/run_port_pp5.sh` -- the image cap follows the budget
  and the drafting reservation instead of pinning the budget to 1024.
* `tests/v1/attention/test_attention_splitting.py` -- short-extend split policy
  and the knob's parsing.
