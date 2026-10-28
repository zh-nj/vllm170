<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## DeepSeek-V4.1-Flash 启动示例

以下示例使用 5 张 GPU，配置 PP5 / TP1、DSpark 5-token 推测解码和 FULL CUDA Graph。
两条命令共用 `26661` 端口，选择其中一条启动；请按实际部署环境调整 GPU 编号、vLLM 路径、模型目录和 SSD 目录。

### DeepSeek-V4.1-Flash-1017：Engram CPU 卸载

使用 `DeepSeek-V4.1-Flash-1017` 模型，启用 Engram CPU 卸载。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3,4,5,6 \
NCCL_P2P_DISABLE=0 \
NCCL_BUFFSIZE=1048576 \
NCCL_MAX_NCHANNELS=8 \
HF_HUB_OFFLINE=1 \
VLLM_PP_LAYER_PARTITION=8,8,8,8,8 \
VLLM_DSV41_HOST_HEADROOM_GIB=12 \
VLLM_DSV41_PREFILL_WORKSPACE_TOKENS=65536 \
VLLM_DEEPSEEK_V4_ENGRAM_GPU_DEQUANT=1 \
VLLM_DEEPSEEK_V4_DECODE_TOPK_CACHE=1 \
VLLM_MARLIN_M_ALIGN=1 \
VLLM_DSV41_ROUTER_GEMM_ALIGN=split \
/mnt/data/apps/vllma100/vllm170/.venv/bin/vllm serve /media/z/data6/models/DeepSeek-V4.1-Flash-1017 \
  --served-model-name dsv41f \
  --host 0.0.0.0 \
  --port 26661 \
  --pipeline-parallel-size 5 \
  --tensor-parallel-size 1 \
  --tokenizer-mode deepseek_v41 \
  --attention-backend TRITON_MLA_SPARSE_DSV41 \
  --moe-backend mxfp4_int8_sm80 \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --default-chat-template-kwargs '{"reasoning_effort":"max"}' \
  --limit-mm-per-prompt '{"image":999}' \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 2700000000 \
  --block-size 128 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","moe_backend":"mxfp4_int8_sm80","enforce_eager":false,"enable_adaptive_verification":false,"dspark_pipeline_stage_ranks":[1,2,3]}' \
  --gpu-memory-utilization 0.98 \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --no-enable-flashinfer-autotune \
  --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --long-prefill-token-threshold 256 \
  --enable-adaptive-long-prefill \
  --engram-config '{"cpu_offload":true,"embedding_across_dp":false}' \
  --additional-config '{"dsv41_sm80_moe_tilelang":true,"dsv41_sm80_fused_swa_metadata":true,"dsv41_sm80_target_prepare_ahead":true,"dsv41_sm80_moe_hybrid_prefill":true,"dsv41_sm80_moe_small_tail_prefill":true,"ced_prefill":true}'
```

### DeepSeek-V4.1-Flash：Engram SSD 卸载

使用 `DeepSeek-V4.1-Flash` 模型，将 Engram 权重卸载到 `/media/z/data6/vllm-engram-ssd`，
scale 保留在内存。此配置将应用层 CPU 行缓存设为 `0.0` GiB，并启用批量读取和预取。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3,4,5,6 \
NCCL_P2P_DISABLE=0 \
NCCL_BUFFSIZE=1048576 \
NCCL_MAX_NCHANNELS=8 \
HF_HUB_OFFLINE=1 \
VLLM_PP_LAYER_PARTITION=8,8,8,8,8 \
VLLM_DSV41_HOST_HEADROOM_GIB=12 \
VLLM_DSV41_PREFILL_WORKSPACE_TOKENS=65536 \
VLLM_DEEPSEEK_V4_ENGRAM_GPU_DEQUANT=1 \
VLLM_DEEPSEEK_V4_DECODE_TOPK_CACHE=1 \
VLLM_MARLIN_M_ALIGN=1 \
VLLM_DSV41_ROUTER_GEMM_ALIGN=split \
/mnt/data/apps/vllma100/vllm170/.venv/bin/vllm serve /media/z/data6/models/DeepSeek-V4.1-Flash \
  --served-model-name dsv41f \
  --host 0.0.0.0 \
  --port 26661 \
  --pipeline-parallel-size 5 \
  --tensor-parallel-size 1 \
  --tokenizer-mode deepseek_v41 \
  --attention-backend TRITON_MLA_SPARSE_DSV41 \
  --moe-backend mxfp4_int8_sm80 \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --default-chat-template-kwargs '{"reasoning_effort":"max"}' \
  --limit-mm-per-prompt '{"image":999}' \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 2700000000 \
  --block-size 128 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","moe_backend":"mxfp4_int8_sm80","enforce_eager":false,"enable_adaptive_verification":false,"dspark_pipeline_stage_ranks":[1,2,3]}' \
  --gpu-memory-utilization 0.98 \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --no-enable-flashinfer-autotune \
  --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --long-prefill-token-threshold 256 \
  --enable-adaptive-long-prefill \
  --engram-config '{"cpu_offload":true,"embedding_across_dp":false,"ssd_offload_path":"/media/z/data6/vllm-engram-ssd","cpu_cache_gb":0.0,"ssd_scale_in_ram":true,"ssd_io_workers":8,"ssd_batch_reads":true,"ssd_prefetch":true}' \
  --additional-config '{"dsv41_sm80_moe_tilelang":true,"dsv41_sm80_fused_swa_metadata":true,"dsv41_sm80_target_prepare_ahead":true,"dsv41_sm80_moe_hybrid_prefill":true,"dsv41_sm80_moe_small_tail_prefill":true,"ced_prefill":true}'
```

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
