"""
web_client/pipeline_loader.py — Pipeline 单例加载与帧解码辅助。
"""
import sys
import threading
from functools import lru_cache
from pathlib import Path

# 确保 web_client 模块在路径中
_web_client_dir = Path(__file__).parent
_project_root = _web_client_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline
from quantizer import apply_fp8_quantization
from utils.misc import set_seed
from utils.wan_wrapper import create_vae_from_config
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from wan.modules.helios_kernels import replace_all_norms_with_flash_norms, replace_rope_with_flash_rope

from web_client.config import LOW_MEMORY_THRESHOLD_GB

_init_lock = threading.Lock()


class FrameCapture:
    """Implements the append_data interface expected by decode_block_and_write,
    capturing numpy frames in memory instead of writing to a file."""
    def __init__(self):
        self.frames = []

    def append_data(self, frame):
        self.frames.append(frame)


_CONFIG_YAML = "configs/long_forcing_dmd.yaml"


def get_pipeline(
    vae_type: str | None = None,
    use_fp8_gemm: bool = False,
    quant_type: str | None = None,
):
    """加载并缓存推理 pipeline（进程内单例）。

    Args:
        vae_type: VAE 模型类型 (wan2.2 | taew2_2 | mg_lightvae | mg_lightvae_v2)。
            若为 ``None``，则从 ``configs/default_config.yaml`` 的 ``vae_type`` 字段读取。
    """
    # Resolve vae_type from default config if not explicitly provided
    if vae_type is None:
        try:
            _dc = OmegaConf.load("configs/default_config.yaml")
            vae_type = str(getattr(_dc, "vae_type", "")).strip().lower() or "taew2_2"
        except Exception:
            vae_type = "taew2_2"
    return _get_pipeline_cached(
        vae_type, use_fp8_gemm, quant_type,
    )


@lru_cache(maxsize=32)
def _get_pipeline_cached(
    vae_type: str,
    use_fp8_gemm: bool,
    quant_type: str | None,
):
    yaml_path = _CONFIG_YAML

    device = torch.device("cuda")
    set_seed(42)
    torch.set_grad_enabled(False)

    free_vram = get_cuda_free_memory_gb(gpu)
    low_memory = free_vram < LOW_MEMORY_THRESHOLD_GB

    config = OmegaConf.load(yaml_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # Ensure config has the resolved vae_type so create_vae_from_config works
    config.vae_type = vae_type

    vae = create_vae_from_config(config)

    pipeline = CausalInferencePipeline(config, device=device, vae=vae)

    replace_all_norms_with_flash_norms(pipeline.generator.model)
    replace_rope_with_flash_rope()
    pipeline = pipeline.to(dtype=torch.bfloat16)

    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)
    if getattr(pipeline, "encoder", None) is not None:
        pipeline.encoder.to(device=gpu)
        print(
            f"[INIT] pipeline.encoder -> {gpu} "
            f"(param_device={next(pipeline.encoder.parameters()).device})",
            flush=True,
        )

    if use_fp8_gemm:
        final_quant_type = str(quant_type) if quant_type is not None else str(
            getattr(config, "quant_type", "fp8-per-token")
        )
        apply_fp8_quantization(
            model=pipeline.generator.model,
            quant_type=final_quant_type,
        )

    return pipeline, config, device


def decode_block_to_frames(pipeline, lat_block) -> list:
    capture = FrameCapture()
    pipeline.decode_block_and_write(lat_block, capture)
    return capture.frames
