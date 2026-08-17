# DeepSeek V4 SM80 PP3 1M context capacity

## Result

Three CMP170HX GPUs can serve one DeepSeek-V4-Flash request at the model's
1,048,576-token deployment limit with PP partition `15,16,12`. The validated
preset is:

```text
VLLM_PP_LAYER_PARTITION=15,16,12
DSV4_MAX_MODEL_LEN=1048576
DSV4_KV_CACHE_BYTES=2700000000
DSV4_MAX_NUM_SEQS=1
DSV4_MAX_BATCHED_TOKENS=1715
DSV4_DSPARK_TOKENS=7
```

Use `benchmarks/deepseek_v4/run_cmp170hx_pp3_1m.sh` to apply these overrides
without changing the existing 131K performance preset.

## Why the partition works

The 43 target layers are split as follows:

| PP rank | Target layers | C4 | C128 | Other responsibility |
| --- | ---: | ---: | ---: | --- |
| PP0 | 0-14 | 7 | 6 | embeddings and two SWA-only layers |
| PP1 | 15-30 | 8 | 8 | highest target-layer residency |
| PP2 | 31-42 | 6 | 6 | final norm, LM head, DSpark draft |

Moving layer 30 from PP2 to PP1 balances the heavy last-stage LM-head and
DSpark residency. At `max_model_len=1M`, model loading used 52.10/54.25/53.24
GiB on PP0/PP1/PP2. PP1 is therefore the capacity-critical stage.

The current FP8 hybrid KV layout needs 2.46 GiB on PP0 and PP1 for one full
request. A 1.6 GB cache fails closed with an estimated maximum length of
350,208. A 2.7 GB cache allocates 2.51 GiB per rank and exposes 1,071,031
tokens, or 1.02 full-length requests.

## Indexer workspace bound

The original sparse-indexer prefill workspace always reserved 40 full-length
histories. That bound came from matching an older FlashMLA workspace budget;
it did not account for scheduler concurrency. A prefill batch cannot contain
more than `max_num_seqs` request histories, so the safe entry bound is now:

```text
max_model_len * min(max_num_seqs, 40)
```

This does not change FP8 cache storage, logits, top-k, or any CUDA kernel. For
the single-sequence 1M preset, the C4 gather reservation falls from 10,485,760
to 262,144 compressed entries. The theoretical FP8 values-plus-scale reduction
is 1,287 MiB; workspace overlay makes the observed process-residency reduction
1,072 MiB on every rank.

| Configuration | PP0 MiB | PP1 MiB | PP2 MiB |
| --- | ---: | ---: | ---: |
| Original 40x bound, ready | 59,216 | 61,852 | 60,256 |
| Scheduler-bounded, ready | 58,144 | 60,780 | 59,184 |
| Difference | -1,072 | -1,072 | -1,072 |

## Real-hardware validation

All requests used exact server-side `/tokenize` calibration and generated
eight tokens through DSpark-7.

| Prompt tokens | TTFT s | Total s | Result |
| ---: | ---: | ---: | --- |
| 32,000 | 6.214 | 6.334 | pass |
| 262,144 | 65.007 | 65.060 | pass |
| 1,048,512 | 562.819 | 562.855 | pass |

The final request used 1,048,520 total prompt-plus-output tokens, below the
1,048,576 model limit. Its sampled PP0/PP1/PP2 high-water marks were
59,892/62,474/60,774 MiB. No CUDA OOM, Xid, or worker restart occurred; the
critical PP1 retained about 2.49 GiB of physical headroom.

These are capacity checks, not production throughput claims. The 1M prefill
rate drops to about 1.86K token/s because C4 indexer scoring repeatedly scans
the growing compressed history. A subsequent row-wave optimization promoted a
96 MiB caller-owned logits workspace with a 1,024-row cap: normalized 262K
TTFT improved by 1.62% without increasing sampled GPU peak memory. Exact
streaming top-k remains experimental because its current combination route
regressed normalized 262K TTFT; the next boundary is score plus block-local
selection fusion.

## Gates and trade-offs

- Keep `max_num_seqs=1` for the validated preset. Higher concurrency needs a
  fresh KV admission calculation and workspace A/B.
- Keep the 2.7 GB KV budget; 1.6 GB is not a 1M configuration.
- Compact MXFP4 indexer KV would save roughly 128 MiB on the eight-C4 PP1
  stage at 1M, but the SM80 logits backend is not available and this is not
  required for capacity.
- Do not add persistent BF16 attention weights. The measured 2.49 GiB PP1
  headroom is capacity reserve, not an optimization budget.
