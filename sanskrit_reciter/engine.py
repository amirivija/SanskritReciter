"""Fully local Sanskrit chant engine (Vāgdhenu pipeline, no network at runtime).

Loads DiT + Vocos + BigVGAN once, then synthesizes WAV from Sanskrit text.
Supports CUDA, Apple MPS, and CPU.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sanskrit_reciter.paths import (
    BANK_JSON,
    BIGVGAN_BASE,
    BIGVGAN_ROOT,
    REFERENCE_BANK,
    VAGDHENU_SRC,
    VOC_CKPT,
    VOCAB_TXT,
    VOCOS_DIR,
    VOICE_CKPT,
    VOICE_FALLBACK,
    pick_device,
    require_assets,
)

SR = 24000

# MPS fallback for ops not yet implemented on Metal
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Never phone home during inference
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _ensure_vendor_on_path() -> None:
    for p in (str(VAGDHENU_SRC), str(BIGVGAN_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _patch_torch_cuda_empty_cache() -> None:
    """IndicF5's load_checkpoint always calls torch.cuda.empty_cache() — guard it."""
    if not hasattr(torch, "cuda"):
        return
    orig = torch.cuda.empty_cache

    def _safe():
        try:
            if torch.cuda.is_available():
                orig()
        except Exception:
            pass

    torch.cuda.empty_cache = _safe  # type: ignore[method-assign]


def _patch_torchaudio_load() -> None:
    """Prefer soundfile when torchcodec is unavailable (pure-local WAV I/O)."""
    import torchaudio

    try:
        # If torchcodec works, leave stock load alone.
        import torchcodec  # noqa: F401
        return
    except Exception:
        pass

    def _load(uri, *args, **kwargs):
        import soundfile as sf

        data, sr = sf.read(str(uri), always_2d=True, dtype="float32")
        # soundfile: [T, C] → torchaudio: [C, T]
        wav = torch.from_numpy(data.T.copy())
        return wav, int(sr)

    torchaudio.load = _load  # type: ignore[assignment]


def _load_vocos_local(device: str):
    """Load Vocos purely from models/vocos-mel-24khz (no HF download)."""
    from vocos import Vocos
    from vocos.feature_extractors import EncodecFeatures

    config_path = VOCOS_DIR / "config.yaml"
    model_path = VOCOS_DIR / "pytorch_model.bin"
    vocoder = Vocos.from_hparams(str(config_path))
    state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in vocoder.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)
    vocoder.load_state_dict(state_dict)
    return vocoder.eval().to(device)


class LocalEngine:
    """Singleton-style renderer: load once, synthesize many verses offline."""

    def __init__(
        self,
        device: str | None = None,
        nfe: int = 32,
        speed: float = 0.90,
        cfg: float = 3.0,
        gap: float = 0.55,
    ):
        missing = require_assets()
        if missing:
            raise FileNotFoundError(
                "Local assets missing. Run once (online):\n"
                "  python scripts/download_models.py\n"
                "  # and ensure vendor/vagdhenu + vendor/BigVGAN are present\n\n"
                "Missing:\n" + "\n".join(missing)
            )

        _ensure_vendor_on_path()
        _patch_torch_cuda_empty_cache()
        _patch_torchaudio_load()

        self.device = pick_device(device)
        self.nfe = nfe
        self.speed = speed
        self.cfg = cfg
        self.gap = gap
        self._renderer = None
        self._alias: dict[str, str] = {}
        self._fallback = "vasantatilakā"

        self._load()

    def _voice_path(self) -> Path:
        if VOICE_CKPT.is_file():
            return VOICE_CKPT
        if VOICE_FALLBACK.is_file():
            return VOICE_FALLBACK
        raise FileNotFoundError("No voice checkpoint under models/vagdhenu/")

    def _load(self) -> None:
        import bigvgan  # vendor/BigVGAN
        from f5_tts.infer.utils_infer import load_model, preprocess_ref_audio_text
        from f5_tts.model import DiT
        from render_core import (  # type: ignore
            Renderer,
            detect_meter_key,
        )

        self._detect_meter_key = detect_meter_key

        # Monkey-patch load_vocoder so Renderer / any side path never hits the network.
        import f5_tts.infer.utils_infer as ui

        def _local_load_vocoder(
            vocoder_name="vocos",
            is_local=False,
            local_path="",
            device=self.device,
            hf_cache_dir=None,
        ):
            if vocoder_name == "vocos":
                return _load_vocos_local(device)
            # BigVGAN path (not used by Vāgdhenu's Capturing vocoder, but keep safe)
            g = bigvgan.BigVGAN.from_pretrained(
                str(BIGVGAN_BASE), use_cuda_kernel=False, local_files_only=True
            )
            g.remove_weight_norm()
            return g.eval().to(device)

        ui.load_vocoder = _local_load_vocoder

        # Build a Renderer subclass that loads BigVGAN from disk (no HF).
        voice = str(self._voice_path())
        voc = str(VOC_CKPT)
        bank = str(BANK_JSON)
        vocab = str(VOCAB_TXT)

        print(f"[engine] device={self.device}  voice={Path(voice).name}", flush=True)

        # Inline init mirroring render_core.Renderer but fully local.
        CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        cfm = load_model(DiT, CFG, mel_spec_type="vocos", vocab_file=vocab, device=self.device)
        ck = torch.load(voice, map_location="cpu", weights_only=True)
        ema = {
            k.replace("ema_model.", ""): v
            for k, v in ck["ema_model_state_dict"].items()
            if k not in ("initted", "step")
        }
        cfm.load_state_dict(ema, strict=False)
        cfm.eval()
        del ck

        real_voc = _load_vocos_local(self.device)

        class Cap:
            def __init__(s, r):
                s.r = r
                s.last = None

            def decode(s, m):
                s.last = m.detach().cpu().numpy()
                return s.r.decode(m)

        cap = Cap(real_voc)

        # Base BigVGAN architecture from local dir, then Vāgdhenu fine-tuned weights.
        g = bigvgan.BigVGAN.from_pretrained(
            str(BIGVGAN_BASE),
            use_cuda_kernel=False,
            local_files_only=True,
            map_location="cpu",
        )
        bsd = torch.load(voc, map_location="cpu")
        bsd = bsd.get("model", bsd)
        g.load_state_dict(bsd)
        try:
            g.remove_weight_norm()
        except Exception:
            pass
        g = g.to(self.device).eval()
        for p in g.parameters():
            p.requires_grad = False

        # Attach state onto a thin Renderer shell to reuse render_one logic.
        # We construct via __new__ and fill fields expected by render_one.
        r = Renderer.__new__(Renderer)
        r.device = self.device
        r.speed = self.speed
        r.nfe = self.nfe
        r.cfg = self.cfg
        r.gap = self.gap
        r.gap_halant = 0.20
        r._preprocess = preprocess_ref_audio_text
        import torchaudio as ta

        r._ta = ta
        r.cfm = cfm
        r.cap = cap
        r.g = g

        def _bvgan(mel):
            m = torch.from_numpy(np.asarray(mel)).to(self.device)
            with torch.no_grad():
                if m.dim() == 3 and m.shape[1] != 100 and m.shape[2] == 100:
                    m = m.transpose(1, 2)
                if m.dim() == 2:
                    # [100, T] or [T, 100]
                    if m.shape[0] == 100:
                        m = m.unsqueeze(0)
                    elif m.shape[1] == 100:
                        m = m.transpose(0, 1).unsqueeze(0)
                return g(m).squeeze().detach().cpu().numpy().astype(np.float32)

        r._bvgan = _bvgan

        bank_data = json.loads(Path(bank).read_text(encoding="utf-8"))
        r._bank = bank_data
        r._bdir = str(REFERENCE_BANK)
        r._lut = {}
        for _k, _v in bank_data.items():
            if _k.startswith("_") or not isinstance(_v, dict) or "wav" not in _v:
                continue
            r._lut[_k.lower()] = _v
            r._lut[_v["wav"].replace(".wav", "").lower()] = _v
            self._alias[_k.lower()] = _k
            self._alias[_v["wav"].replace(".wav", "").lower()] = _k
        r._primes = bank_data.get("repeat_primes", {})
        r._refcache = {}

        # Bind methods from the class that render_one needs
        r._get_ref = Renderer._get_ref.__get__(r, Renderer)
        r._stitch = Renderer._stitch.__get__(r, Renderer)
        r.render_one = Renderer.render_one.__get__(r, Renderer)
        r.meters = Renderer.meters.__get__(r, Renderer)

        self._renderer = r

        if "vasantatilakā" in bank_data:
            self._fallback = "vasantatilakā"
        elif "vasantatilaka" in self._alias:
            self._fallback = self._alias["vasantatilaka"]
        else:
            meters = [
                k
                for k, v in bank_data.items()
                if not k.startswith("_") and isinstance(v, dict) and "wav" in v
            ]
            if meters:
                self._fallback = meters[0]

        print("[engine] models loaded (offline)", flush=True)

    def resolve_meter(self, text: str, meter: str = "__auto__") -> tuple[str, str]:
        if not meter or meter in ("__auto__", "auto"):
            key = self._detect_meter_key(text)
            resolved = self._alias.get((key or "").lower())
            if resolved:
                return resolved, f"Detected meter: {resolved}"
            return self._fallback, f"Meter unknown — using {self._fallback}"
        resolved = self._alias.get(meter.lower(), meter)
        return resolved, f"Meter: {resolved}"

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        *,
        meter: str = "__auto__",
        seed: int = 60,
    ) -> tuple[int, np.ndarray, str]:
        """Return (sample_rate, float32 audio, status)."""
        from sanskrit_reciter.text_io import clean_verse, is_real_verse

        text = clean_verse(text or "")
        if not is_real_verse(text):
            raise ValueError(
                f"Not a synthesizable Sanskrit verse (too short or only numbers/notes): {text!r}"
            )

        used, status = self.resolve_meter(text, meter)
        print(f"[engine] {status}", flush=True)

        # seed on all backends
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sr, audio = self._renderer.render_one(text, used, seed=int(seed))
        except IndexError as e:
            # F5 infer_batch_process returns empty waves when gen text is empty after G2P
            raise RuntimeError(
                f"Synthesis produced no audio (empty model input?). verse={text!r}"
            ) from e

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError(f"Synthesis returned empty audio for: {text!r}")
        # peak normalize softly
        peak = float(np.abs(audio).max())
        if peak > 1.0:
            audio = audio / peak * 0.97
        return int(sr), audio, status

    def synthesize_to_file(
        self,
        text: str,
        out_path: str | Path,
        *,
        meter: str = "__auto__",
        seed: int = 60,
    ) -> tuple[Path, str]:
        sr, audio, status = self.synthesize(text, meter=meter, seed=seed)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, sr)
        return out, status


# Module-level cache so CLI multi-verse runs only load once
_ENGINE: LocalEngine | None = None


def get_engine(**kwargs) -> LocalEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = LocalEngine(**kwargs)
    return _ENGINE
