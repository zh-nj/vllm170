# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from pydantic import Field

import vllm.envs as envs
from vllm.config.utils import config, get_hash_factors, hash_factors
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

logger = init_logger(__name__)

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
    "Qwen4ExpForCausalLM": "ple_layer_ids",
    "Qwen4ExpForConditionalGeneration": "ple_layer_ids",
}


def _default_cpu_offload() -> bool:
    """Honor the legacy environment variable only when the field is omitted."""
    if envs.is_set("VLLM_PLE_CPU_OFFLOAD"):
        logger.warning_once(
            "VLLM_PLE_CPU_OFFLOAD is a legacy setting and may be removed in a "
            "future release. Use --engram-config.cpu_offload instead."
        )
    return envs.VLLM_PLE_CPU_OFFLOAD


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = Field(default_factory=_default_cpu_offload)
    """Store embedding weights in pinned CPU memory for UVA lookup.
    Defaults to False, or VLLM_PLE_CPU_OFFLOAD when set for compatibility.
    An explicit value takes precedence over the legacy environment variable.
    With SSD storage, host memory holds the configured scales, cache and staging."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    cpu_cache_gb: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    """Capacity in GiB of each SSD-backed table's pinned CPU FIFO cache.
    Cache hits keep their original insertion order; misses replace the oldest
    resident rows. A nonzero budget requires ``ssd_offload_path``."""

    ssd_offload_path: str | None = Field(default=None, min_length=1)
    """Directory for complete file-backed Engram tables.
    The directory should be on a local SSD and is created when necessary."""

    ssd_scale_in_ram: bool = True
    """Keep SSD table scales in pageable CPU RAM, outside the row-cache budget.
    Weight bytes remain on SSD. Disable to minimize resident host memory."""

    ssd_io_workers: int = Field(default=8, ge=1, le=32)
    """Maximum concurrent SSD row readers per local table; 1 reads serially.
    Small requests and fast page-cache reads use the serial path.
    This field has no effect unless ``ssd_offload_path`` is set."""

    ssd_batch_reads: bool = False
    """Use batched mapped reads for large, page-cache-resident SSD requests.
    Slow probe reads retain the parallel preadv path. Requires RAM scales."""

    ssd_prefetch: bool = False
    """Start same-step SSD row reads before receiving upstream PP tensors.
    GPU staging remains at the original lookup boundary. Requires V2 PP."""

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        from vllm.platforms import current_platform

        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not current_platform.is_cuda()
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings. Currently only the CUDA Qwen4Exp and DeepSeek "
                "V4.1 implementations with non-empty n-gram layer ids are "
                "supported."
            )
        if self.ssd_offload_path and not self.cpu_offload:
            raise ValueError("Engram ssd_offload_path requires cpu_offload")
        if self.ssd_offload_path and field != "engram_layer_ids":
            raise ValueError("Engram SSD storage currently requires DeepSeek V4.1")
        if self.cpu_cache_gb and not self.ssd_offload_path:
            raise ValueError("Engram cpu_cache_gb requires ssd_offload_path")

    def verify_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject unsupported embedding parallel topologies."""
        if (
            self.ssd_offload_path
            and self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
        ):
            raise ValueError("Engram SSD storage does not support sharding across DP")
        if (
            self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
            and parallel_config.enable_elastic_ep
        ):
            raise ValueError(
                "Engram embedding_across_dp is not supported with elastic EP yet."
            )

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
