"""Project paths for fully local assets (weights, vendor code, reference bank)."""

from __future__ import annotations

import os
from pathlib import Path

# Package lives at <root>/sanskrit_reciter/
ROOT = Path(__file__).resolve().parents[1]

# Vendor trees (cloned once by setup)
VAGDHENU_SRC = ROOT / "vendor" / "vagdhenu" / "src"
BIGVGAN_ROOT = ROOT / "vendor" / "BigVGAN"
REFERENCE_BANK = VAGDHENU_SRC / "reference_bank"
BANK_JSON = REFERENCE_BANK / "bank.json"

# Weights downloaded by scripts/download_models.py
MODELS = Path(os.environ.get("SANSKRIT_RECITER_MODELS", ROOT / "models"))
VOICE_CKPT = MODELS / "vagdhenu" / "voice_steer_ema_2026-06-17.pt"
VOICE_FALLBACK = MODELS / "vagdhenu" / "voice_armA_ema_2026-06-11.pt"
VOC_CKPT = MODELS / "vagdhenu" / "voc_bigvgan_EMA_2026-06-11.pth"
VOCAB_TXT = MODELS / "vagdhenu" / "vocab.txt"
# Prefer bank-bundled vocab if present (same file)
if (REFERENCE_BANK / "vocab.txt").is_file():
    VOCAB_TXT = REFERENCE_BANK / "vocab.txt"

BIGVGAN_BASE = MODELS / "bigvgan_v2_24khz_100band_256x"
VOCOS_DIR = MODELS / "vocos-mel-24khz"


def pick_device(requested: str | None = None) -> str:
    """cuda → mps → cpu, or an explicit override."""
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def require_assets() -> list[str]:
    """Return human-readable list of missing required files (empty if ready)."""
    missing = []
    checks = {
        "Vāgdhenu src": VAGDHENU_SRC / "render_core.py",
        "BigVGAN code": BIGVGAN_ROOT / "bigvgan.py",
        "reference bank": BANK_JSON,
        "voice checkpoint": VOICE_CKPT if VOICE_CKPT.is_file() else VOICE_FALLBACK,
        "BigVGAN EMA vocoder": VOC_CKPT,
        "vocab.txt": VOCAB_TXT,
        "BigVGAN base dir": BIGVGAN_BASE / "config.json",
        "Vocos config": VOCOS_DIR / "config.yaml",
        "Vocos weights": VOCOS_DIR / "pytorch_model.bin",
    }
    for label, path in checks.items():
        if not Path(path).exists():
            missing.append(f"  • {label}: {path}")
    return missing
