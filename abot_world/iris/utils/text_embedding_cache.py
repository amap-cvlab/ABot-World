"""Optional UMT5 rows for the exact encoder, tokenizer and texts in use."""

import hashlib
import logging
from pathlib import Path

DEFAULT_TEXT_CACHE = "outputs/text_cache/abot-example-text-cache.pt"
CACHE_VERSION = 1
TOKEN_LENGTHS = {"global": 384, "local": 128}
SOURCE_ROOT = Path(__file__).resolve().parents[1]
ENCODER_SOURCES = ("utils/wan_wrapper.py", "wan/modules/t5.py", "wan/modules/tokenizers.py")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_encoder_identity(encoder_pth_path, tokenizer_path):
    tokenizer = Path(tokenizer_path)
    files = sorted(p for p in tokenizer.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"No tokenizer files found in {tokenizer}")
    return {
        "encoder": _sha256(Path(encoder_pth_path)),
        "tokenizer": {p.relative_to(tokenizer).as_posix(): _sha256(p) for p in files},
        "source": {rel: _sha256(SOURCE_ROOT / rel) for rel in ENCODER_SOURCES},
    }


def load_text_cache(path, encoder_kwargs, dimension):
    """Return BF16 CPU rows or fall back to ordinary text encoding."""
    import torch

    path = Path(path)
    if not path.is_file():
        return None
    try:
        cache = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if cache["version"] != CACHE_VERSION or cache["identity"] != text_encoder_identity(**encoder_kwargs):
            raise ValueError("cache was generated with a different encoder, tokenizer or encoding implementation")
        for name, length in TOKEN_LENGTHS.items():
            if not isinstance(cache[name], dict) or not cache[name]:
                raise ValueError(f"Missing {name} rows")
            for text, row in cache[name].items():
                if (not isinstance(text, str) or not isinstance(row, torch.Tensor)
                        or row.shape != (1, length, dimension) or row.dtype != torch.bfloat16
                        or not torch.isfinite(row).all()):
                    raise ValueError(f"Invalid {name} embedding row")
        return cache
    except Exception as exc:
        logging.getLogger(__name__).warning("Ignoring text cache %s: %s", path, exc)
        return None
