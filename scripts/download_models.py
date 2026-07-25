#!/usr/bin/env python3
"""One-time download of all weights needed for fully offline Sanskrit recitation.

Run while online:
    python scripts/download_models.py

After this, inference uses only files under models/ (no network).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# Keep HF cache inside the project so the whole tree is portable.
os.environ.setdefault("HF_HOME", str(MODELS / "hf_home"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS / "hf_home" / "hub"))


def main() -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    print(f"→ models dir: {MODELS}")

    # 1) Vāgdhenu voice + fine-tuned BigVGAN EMA + vocab
    print("\n[1/3] prathoshap/vagdhenu (voice + chant vocoder + vocab)")
    for f in [
        "voice_steer_ema_2026-06-17.pt",
        "voice_armA_ema_2026-06-11.pt",
        "voc_bigvgan_EMA_2026-06-11.pth",
        "vocab.txt",
    ]:
        path = hf_hub_download(
            "prathoshap/vagdhenu",
            f,
            local_dir=str(MODELS / "vagdhenu"),
        )
        print("  ✓", path)

    # 2) NVIDIA BigVGAN base weights (loaded, then overridden by Vāgdhenu EMA)
    print("\n[2/3] nvidia/bigvgan_v2_24khz_100band_256x")
    bv = snapshot_download(
        "nvidia/bigvgan_v2_24khz_100band_256x",
        local_dir=str(MODELS / "bigvgan_v2_24khz_100band_256x"),
        ignore_patterns=["*.md", "*.png", "*.jpg", "samples/*"],
    )
    print("  ✓", bv)

    # 3) Vocos (mel frontend used during F5 infer_process; BigVGAN re-vocodes the mel)
    print("\n[3/3] charactr/vocos-mel-24khz")
    for f in ["config.yaml", "pytorch_model.bin"]:
        path = hf_hub_download(
            "charactr/vocos-mel-24khz",
            f,
            local_dir=str(MODELS / "vocos-mel-24khz"),
        )
        print("  ✓", path)

    print("\n✓ All weights ready under", MODELS)
    print("  You can now run offline:")
    print("    HF_HUB_OFFLINE=1 python -m sanskrit_reciter examples/krishna.txt -o out/krishna.wav")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"download failed: {e}", file=sys.stderr)
        raise
