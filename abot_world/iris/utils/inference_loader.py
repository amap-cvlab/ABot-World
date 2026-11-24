"""Construct the standalone streaming model, without application services."""
import torch
from abot_world.iris.config import load_iris_config
from abot_world.iris.pipeline.causal_inference import CausalInferencePipeline
from abot_world.iris.utils.misc import set_seed
from abot_world.iris.utils.memory import DynamicSwapInstaller
from abot_world.iris.utils.runtime_memory import validate_quantization_device
from abot_world.iris.utils.wan_wrapper import create_vae_from_config
from abot_world.iris.wan.modules.helios_kernels import replace_all_norms_with_flash_norms, replace_rope_with_flash_rope


def build_pipeline(config, device):
    pipeline = CausalInferencePipeline(config, device=device, vae=create_vae_from_config(config))
    replace_all_norms_with_flash_norms(pipeline.generator.model)
    replace_rope_with_flash_rope()
    return pipeline.to(dtype=torch.bfloat16)


def load_streaming_model(config_path):
    config = load_iris_config(config_path)
    device = torch.device('cuda')
    validate_quantization_device(config, device)
    set_seed(int(config.get('seed', 42)))
    checkpoint = config.get('inference_checkpoint')
    if checkpoint:
        config.model_kwargs.inference_checkpoint = checkpoint
        config.model_kwargs.inference_use_ema = bool(config.get('inference_use_ema', False))
    pipeline = build_pipeline(config, device)
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    if config.get('use_fp8_gemm', True):
        from abot_world.iris.quantizer import apply_fp8_quantization

        apply_fp8_quantization(pipeline.generator.model, quant_type=config.get('quant_type', 'fp8-per-token'))
    pipeline.eval().requires_grad_(False)
    return pipeline, config, device
