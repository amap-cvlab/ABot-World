"""
Iris pipeline caching and in-memory frame decoding for interactive clients.
"""
import threading
from functools import lru_cache

import torch
from abot_world.iris.utils.inference_loader import build_pipeline
from abot_world.iris.utils.misc import set_seed
from abot_world.iris.utils.runtime_memory import validate_quantization_device
from abot_world.iris.utils.text_embedding_cache import DEFAULT_TEXT_CACHE, load_text_cache
from abot_world.iris.utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from abot_world.iris.config import LOW_MEMORY_THRESHOLD_GB, load_interactive_config

_init_lock = threading.Lock()


class FrameCapture:
    """Implements the append_data interface expected by decode_block_and_write,
    capturing numpy frames in memory instead of writing to a file."""
    def __init__(self):
        self.frames = []

    def append_data(self, frame):
        self.frames.append(frame)


def get_pipeline(
    vae_type: str | None = None,
    use_fp8_gemm: bool = False,
    quant_type: str | None = None,
):
    """加载并缓存推理 pipeline（进程内单例）。

    Args:
        vae_type: Decoder backend. If unset, use the merged demo configuration.
    """
    if vae_type is None:
        vae_type = str(load_interactive_config().get("vae_type", "taew2_2")).strip().lower()
    return _get_pipeline_cached(
        vae_type, use_fp8_gemm, quant_type,
    )


@lru_cache(maxsize=32)
def _get_pipeline_cached(
    vae_type: str,
    use_fp8_gemm: bool,
    quant_type: str | None,
):
    device = torch.device("cuda")
    set_seed(42)
    torch.set_grad_enabled(False)

    free_vram = get_cuda_free_memory_gb(gpu)
    low_memory = free_vram < LOW_MEMORY_THRESHOLD_GB

    config = load_interactive_config()

    # Ensure config has the resolved vae_type so create_vae_from_config works
    config.vae_type = vae_type
    config.use_fp8_gemm = use_fp8_gemm
    if quant_type is not None:
        config.quant_type = quant_type
    validate_quantization_device(config, device)

    pipeline = build_pipeline(config, device)

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
        from abot_world.iris.quantizer import apply_fp8_quantization

        final_quant_type = str(quant_type) if quant_type is not None else str(
            getattr(config, "quant_type", "fp8-per-token")
        )
        apply_fp8_quantization(
            model=pipeline.generator.model,
            quant_type=final_quant_type,
        )

    cache = load_text_cache(DEFAULT_TEXT_CACHE, dict(config.text_encoder_kwargs),
                            pipeline.generator.model.text_dim)
    if cache is not None:
        pipeline.install_text_cache(cache, device)
        print(f"[INIT] Using pre-encoded text from {DEFAULT_TEXT_CACHE}", flush=True)

    return pipeline, config, device


def decode_block_to_frames(pipeline, lat_block, *, final=False) -> list:
    capture = FrameCapture()
    pipeline.decode_block_and_write(lat_block, capture, final=final)
    return capture.frames
