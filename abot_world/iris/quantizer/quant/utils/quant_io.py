# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, Optional, Union

import torch

__all__ = ["load_fp8_scales", "load_quantized_model", "save_quantized_model"]


def load_fp8_scales(
    quant_scales: Optional[Union[str, Dict[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    """Load FP8 quant scales from dict, file, or dir. Prefer .safetensors."""
    if quant_scales is None:
        raise ValueError("quant_scales is required")

    if isinstance(quant_scales, dict) and len(quant_scales) > 0:
        # Use provided dict
        return quant_scales

    if isinstance(quant_scales, str):
        # Check if path is file
        if os.path.isfile(quant_scales):
            if quant_scales.endswith(".safetensors"):
                import safetensors.torch

                print(f"Loaded scale map from {quant_scales}")
                return safetensors.torch.load_file(quant_scales)
            else:
                print(f"Loaded scale map from {quant_scales}")
                return torch.load(quant_scales)
        # Check if path is directory
        if os.path.isdir(quant_scales):
            safetensors_path = os.path.join(quant_scales, "fp8_scales.safetensors")
            pth_path = os.path.join(quant_scales, "fp8_scales.pth")
            if os.path.isfile(safetensors_path):
                import safetensors.torch

                print(f"Loaded scale map from {safetensors_path}")
                return safetensors.torch.load_file(safetensors_path)
            if os.path.isfile(pth_path):
                print(f"Loaded scale map from {pth_path}")
                return torch.load(pth_path)
            raise FileNotFoundError(
                f"Quant scale file not found: {pth_path} or {safetensors_path}"
            )
        raise FileNotFoundError(f"quant_scales path does not exist: {quant_scales}")

    raise ValueError(f"Invalid quant_scales type: {type(quant_scales)}. Only str (path) or dict.")


def save_quantized_model(model: torch.nn.Module, save_path: str, fp8_scales_map: Dict):
    """
    Save quantized model and scale dict to directory.
    """
    import logging

    logger = logging.getLogger(__name__)

    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Cannot create directory for save_path: {save_path}. Error: {e}")

    try:
        # If Hugging Face style, use save_pretrained
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(save_path)
            logger.info(f"Saved quantized model to {save_path} via save_pretrained")
        else:
            # Otherwise, save state_dict with safetensors
            from safetensors.torch import save_file as safe_save

            model_path = os.path.join(save_path, "model.safetensors")
            safe_save(model.state_dict(), model_path)
            logger.info(f"Saved state_dict to {model_path}")

        # Always save scales dict
        from safetensors.torch import save_file as safe_save

        scale_save_path = os.path.join(save_path, "fp8_scales.safetensors")
        safe_save(fp8_scales_map, scale_save_path)
        logger.info(f"Saved scales map to {scale_save_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to save model and scales map to {save_path}. Error: {e}")


def load_quantized_model(model_class, save_path: str, device: str = "cpu"):
    """
    Load quantized model from directory.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Try Hugging Face style first
        if hasattr(model_class, "from_pretrained"):
            model = model_class.from_pretrained(save_path)
            logger.info(f"Loaded Hugging Face model from {save_path}")
            return model
    except Exception as e:
        logger.warning(f"Failed to load as Hugging Face model: {e}")

    try:
        # Try safetensors file first
        model_path = os.path.join(save_path, "model.safetensors")
        if os.path.exists(model_path):
            from safetensors.torch import load_file as safe_load

            state_dict = safe_load(model_path, device=device)
            model = model_class()
            model.load_state_dict(state_dict)
            model.to(device)
            logger.info(f"Loaded model from {model_path} (safetensors)")
            return model

        # Try pytorch .bin next
        model_path = os.path.join(save_path, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model = model_class()
            model.load_state_dict(state_dict)
            model.to(device)
            logger.info(f"Loaded model from {model_path} (pytorch)")
            return model
        else:
            raise FileNotFoundError(
                f"Model file not found at {save_path}. "
                "Expected 'model.safetensors' or 'pytorch_model.bin'"
            )

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {save_path}. Error: {e}")
