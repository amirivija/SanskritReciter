"""Build a LearningPack from a śloka text file (+ optional annotation sidecar)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import soundfile as sf

from sanskrit_reciter.learning.annotations_io import load_annotations_json, merge_annotations
from sanskrit_reciter.learning.html_render import write_html_pack
from sanskrit_reciter.learning.models import LearningPack, VerseUnit
from sanskrit_reciter.text_io import read_text, split_shlokas


ProgressFn = Callable[[str], None]


def build_learning_pack(
    text_file: str | Path,
    out_dir: str | Path,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    description: str | None = None,
    annotations_file: str | Path | None = None,
    engine=None,
    device: str = "auto",
    nfe: int = 32,
    meter: str = "__auto__",
    seed: int = 60,
    continue_on_error: bool = True,
    skip_audio: bool = False,
    progress: ProgressFn | None = None,
) -> tuple[LearningPack, Path]:
    """Create a learning pack directory with per-verse audio + HTML.

    Returns ``(pack, index_html_path)``.
    """
    log = progress or (lambda msg: print(msg, flush=True))
    text_path = Path(text_file)
    root = Path(out_dir)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    raw = read_text(text_path)
    shlokas = split_shlokas(raw)
    if not shlokas:
        raise ValueError(f"No Sanskrit verses found in {text_path}")

    pack_title = title or text_path.stem.replace("_", " ").title()
    pack = LearningPack(
        title=pack_title,
        subtitle=subtitle,
        description=description,
        meta={
            "source_file": str(text_path.name),
            "verse_count": len(shlokas),
        },
    )

    width = max(3, len(str(len(shlokas))))
    for i, text in enumerate(shlokas, 1):
        vid = f"{i:0{width}d}"
        pack.verses.append(
            VerseUnit(id=vid, text=text, index=i)
        )

    if annotations_file:
        data = load_annotations_json(annotations_file)
        merge_annotations(pack, data)
        log(f"merged annotations from {annotations_file}")

    if not skip_audio:
        if engine is None:
            from sanskrit_reciter.engine import LocalEngine
            from sanskrit_reciter.paths import require_assets

            missing = require_assets()
            if missing:
                raise FileNotFoundError(
                    "Local assets missing:\n" + "\n".join(missing)
                )
            log(f"loading TTS engine (device={device}, nfe={nfe})…")
            engine = LocalEngine(device=device, nfe=nfe)

        log(f"synthesizing {len(pack.verses)} verse(s)…")
        ok = failed = 0
        for v in pack.verses:
            label = f"[{v.index}/{len(pack.verses)}]"
            preview = v.text.replace("\n", " ")[:50]
            log(f"{label} {preview}{'…' if len(v.text) > 50 else ''}")
            try:
                sr, audio, status = engine.synthesize(
                    v.text, meter=meter, seed=seed
                )
                # Prefer resolved meter from status when auto
                if status and "meter" in status.lower():
                    # "Detected meter: anuṣṭubh" / "Meter: …"
                    parts = status.split(":", 1)
                    if len(parts) == 2:
                        v.meter = parts[1].strip()
                rel = f"audio/{v.id}.wav"
                out_wav = root / rel
                sf.write(str(out_wav), audio, int(sr))
                v.audio = rel
                v.error = None
                ok += 1
                log(f"{label} ok → {rel}" + (f"  ({status})" if status else ""))
            except Exception as e:
                failed += 1
                v.error = str(e)
                v.audio = None
                log(f"{label} failed: {e}")
                if not continue_on_error:
                    raise
        pack.meta["audio_ok"] = ok
        pack.meta["audio_failed"] = failed
        log(f"audio done: {ok} ok, {failed} failed")
    else:
        log("skip_audio: writing HTML structure only")

    index = write_html_pack(pack, root)
    log(f"wrote learning pack → {index}")
    return pack, index


def rebuild_html_only(
    out_dir: str | Path,
    *,
    annotations_file: str | Path | None = None,
) -> Path:
    """Re-render HTML from an existing pack manifest (no TTS)."""
    import json

    root = Path(out_dir)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"No manifest.json in {root}")
    pack = LearningPack.from_dict(json.loads(manifest.read_text(encoding="utf-8")))
    if annotations_file:
        merge_annotations(pack, load_annotations_json(annotations_file))
    return write_html_pack(pack, root)
