"""Iris configuration shared by the CLI and Gradio."""

import os
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "iris"
DEFAULT_STREAM_CONFIG = CONFIG_DIR / "model_inference.yaml"
LOW_MEMORY_THRESHOLD_GB = 40
STREAM_HEIGHT = 704
STREAM_WIDTH = 1280
LATENT_CHANNELS = 48
LATENT_HEIGHT = STREAM_HEIGHT // 16
LATENT_WIDTH = STREAM_WIDTH // 16
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VIDEO_FPS = 12
KEY_ORDER = list("WASDIJKL")


def load_iris_config(overlay=None):
    paths = [CONFIG_DIR / "default_config.yaml", CONFIG_DIR / "long_forcing_dmd.yaml"]
    if overlay:
        paths.append(overlay)
    return OmegaConf.merge(*(OmegaConf.load(path) for path in paths))


def load_interactive_config():
    overlay = os.environ.get("ABOTWORLD_CONFIG")
    return load_iris_config(PROJECT_ROOT / overlay if overlay else None)
