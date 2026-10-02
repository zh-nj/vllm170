# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare live V41 metadata builders with individual optimizations on/off."""

import argparse
import hashlib
import json
import statistics
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4.phase_profile import phase_profile_enabled
from vllm.models.deepseek_v4_1 import compressor, sparse_mla
from vllm.models.deepseek_v4_1.amd import rocm
from vllm.models.deepseek_v4_1.amd.rocm import (
    DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


def make_builders(device, *, fused_swa=False):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=2048),
        model_config=SimpleNamespace(
            max_model_len=1048576,
            hf_config=SimpleNamespace(
                sliding_window=128, index_topk=512, compress_ratios=[0, 1, 2]
            ),
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=5,
            parallel_drafting=False,
            use_dspark=lambda: True,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        attention_config=None,
        additional_config={"dsv41_sm80_fused_swa_metadata": fused_swa},
    )
    builders = []
    for ratio in (1, 2):
        spec = MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            cache_dtype_str="fp8_ds_mla",
            tokens_per_state=ratio,
        )
        builders.append(
            sparse_mla.DeepseekV4SparseMLAMetadataBuilder(spec, ["mla"], config, device)
        )
    spec = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=1024, dtype=torch.float32
    )
    builders.append(
        compressor.CompressorMetadataBuilder(spec, ["compressor"], config, device)
    )
    for _ in range(4):
        spec = SlidingWindowMLASpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            sliding_window=128,
            cache_dtype_str="fp8_ds_mla",
        )
        builders.append(
            DeepseekV4ROCMAiterSparseSWAMetadataBuilder(spec, ["swa"], config, device)
        )
    return builders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument(
        "--variant", choices=("token-map", "swa-metadata"), default="token-map"
    )
    args = parser.parse_args()
    if args.samples < 10:
        parser.error("at least 10 samples required")
    if phase_profile_enabled():
        parser.error("phase profiling must be disabled")
    device = torch.device("cuda:0")
    builders = make_builders(device, fused_swa=args.variant == "swa-metadata")
    assert all(b._persist_single_req_token_map for b in builders[:3])
    if args.variant == "swa-metadata":
        assert all(b._use_fused_swa_metadata for b in builders[3:])

    def select(fast):
        if args.variant == "token-map":
            for b in builders[:3]:
                b._persist_single_req_token_map = fast
        else:
            for b in builders[3:]:
                b._use_fused_swa_metadata = fast

    results = []
    # One or two MLA groups plus four SWA groups occur in the PP5 trace.
    rosters = {
        "mla2_compressor": [0, 1, 2],
        "mla1_compressor_swa4": [1, 2, 3, 4, 5, 6],
        "mla2_swa4": [0, 1, 3, 4, 5, 6],
    }
    for reqs, width in ((1, 6), (1, 5), (2, 6)):
        tokens = reqs * width
        cpu_qsl = torch.arange(reqs + 1, dtype=torch.int32) * width
        qsl = cpu_qsl.to(device)
        seq = torch.full((reqs,), 32768, dtype=torch.int32, device=device)
        table = torch.arange(reqs * 8192, dtype=torch.int32, device=device).view(
            reqs, -1
        )
        slots = torch.arange(tokens, dtype=torch.int64, device=device)
        pos = (
            torch.arange(tokens, dtype=torch.int64, device=device).remainder(width)
            + 32768
            - width
        )

        common = partial(
            CommonAttentionMetadata,
            query_start_loc=qsl,
            query_start_loc_cpu=cpu_qsl,
            seq_lens=seq,
            num_reqs=reqs,
            num_actual_tokens=tokens,
            max_query_len=width,
            max_seq_len=32768,
            block_table_tensor=table,
            slot_mapping=slots,
            positions=pos,
            causal=not (args.variant == "swa-metadata" and width == 5),
        )

        for roster, indices in rosters.items():

            def run(indices=tuple(indices), common=common):
                return [builders[i].build(0, common()) for i in indices]

            def maps(outputs):
                return [
                    m.req_id_per_token
                    if hasattr(m, "req_id_per_token")
                    else m.token_to_req_indices
                    for m in outputs
                ]

            def swa_outputs(outputs):
                values = []
                for m in outputs:
                    if not hasattr(m, "decode_swa_ragged_indptr"):
                        continue
                    nnz = int(m.decode_swa_ragged_indptr[-1].item())
                    values.extend(
                        v.clone()
                        for v in (
                            m.is_valid_token,
                            m.decode_swa_lens,
                            m.decode_swa_ragged_indptr,
                            m.decode_swa_ragged_indices[:nnz],
                        )
                    )
                return values

            expected = torch.arange(
                reqs, dtype=torch.int32, device=device
            ).repeat_interleave(width)
            raw = {"baseline": [], "candidate": []}
            gpu_raw = {"baseline": [], "candidate": []}
            graphs = {}
            reference = None
            for name, fast in (("baseline", False), ("candidate", True)):
                select(fast)
                for m in maps(run()):
                    torch.testing.assert_close(m, expected, atol=0, rtol=0)
                for _ in range(10):
                    run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = run()
                graph.replay()
                for m in maps(outputs):
                    torch.testing.assert_close(m, expected, atol=0, rtol=0)
                actual = swa_outputs(outputs)
                if reference is None:
                    reference = actual
                else:
                    for a, b in zip(actual, reference):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
                graphs[name] = graph
            for round_index in range(3):
                names = ["baseline", "candidate"]
                if round_index % 2:
                    names.reverse()
                for name in names:
                    select(name == "candidate")
                    for _ in range(10):
                        run()
                    torch.accelerator.synchronize()
                    host, completed, gpu = [], [], []
                    for _ in range(args.samples):
                        t0 = time.perf_counter_ns()
                        run()
                        t1 = time.perf_counter_ns()
                        torch.accelerator.synchronize()
                        t2 = time.perf_counter_ns()
                        host.append((t1 - t0) / 1e6)
                        completed.append((t2 - t0) / 1e6)
                        start = torch.Event(enable_timing=True)
                        end = torch.Event(enable_timing=True)
                        start.record()
                        graphs[name].replay()
                        end.record()
                        end.synchronize()
                        gpu.append(start.elapsed_time(end))
                    raw[name].append(dict(host_ms=host, completed_ms=completed))
                    gpu_raw[name].append(gpu)
            summary = {
                name: dict(
                    host_p50_ms=statistics.median(
                        sum((r["host_ms"] for r in rounds), [])
                    ),
                    completed_p50_ms=statistics.median(
                        sum((r["completed_ms"] for r in rounds), [])
                    ),
                    graph_gpu_p50_ms=statistics.median(sum(gpu_raw[name], [])),
                    round_host_p50_ms=[statistics.median(r["host_ms"]) for r in rounds],
                )
                for name, rounds in raw.items()
            }
            results.append(
                dict(
                    requests=reqs,
                    width=width,
                    roster=roster,
                    exact_maps=True,
                    exact_swa_outputs=True,
                    summary=summary,
                    raw=raw,
                    graph_gpu_raw=gpu_raw,
                )
            )
            print(reqs, width, roster, summary, flush=True)
    result = dict(
        scope="real builder.build; synthetic 32K inputs; excludes indexer and runner",
        timing="3 alternating rounds; eager host/enqueue + completion; GPU separately",
        phase_enabled=False,
        variant=args.variant,
        gpu=str(torch.cuda.get_device_properties(0)),
        source_sha256={
            m.__file__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (sparse_mla, compressor, rocm)
        },
        results=results,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
