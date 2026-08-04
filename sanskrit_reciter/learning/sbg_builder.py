"""Build & synthesize Bhagavad Gītā learning packs (per chapter) with resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import soundfile as sf

from sanskrit_reciter.learning.html_render import write_html_pack
from sanskrit_reciter.learning.models import LearningPack
from sanskrit_reciter.learning.publish import publish_pack, write_catalog
from sanskrit_reciter.learning.sbg import (
    DEFAULT_DATA_DIR,
    build_chapter_pack,
    load_sbg_data,
)

ProgressFn = Callable[[str], None]


def synthesize_pack(
    pack: LearningPack,
    out_dir: str | Path,
    *,
    engine=None,
    device: str = "auto",
    nfe: int = 32,
    meter: str = "__auto__",
    seed: int = 60,
    continue_on_error: bool = True,
    resume: bool = True,
    skip_audio: bool = False,
    progress: ProgressFn | None = None,
) -> LearningPack:
    """Write pack to out_dir, synthesize missing audio (resume-friendly)."""
    log = progress or (lambda m: print(m, flush=True))
    root = Path(out_dir)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if skip_audio:
        log("skip_audio: writing HTML only")
        write_html_pack(pack, root)
        return pack

    if engine is None:
        from sanskrit_reciter.engine import LocalEngine
        from sanskrit_reciter.paths import require_assets

        missing = require_assets()
        if missing:
            raise FileNotFoundError("Local assets missing:\n" + "\n".join(missing))
        log(f"loading TTS engine (device={device}, nfe={nfe})…")
        engine = LocalEngine(device=device, nfe=nfe)

    ok = skipped = failed = 0
    total = len(pack.verses)
    for v in pack.verses:
        rel = f"audio/{v.id}.wav"
        dest = root / rel
        label = f"[{v.index}/{total}] {v.source_ref or v.id}"

        if resume and dest.is_file() and dest.stat().st_size > 1000:
            v.audio = rel
            v.error = None
            skipped += 1
            log(f"{label} resume → {rel}")
            continue

        preview = v.text.replace("\n", " ")[:48]
        log(f"{label} synthesizing… {preview}…")
        try:
            sr, audio, status = engine.synthesize(v.text, meter=meter, seed=seed)
            sf.write(str(dest), audio, int(sr))
            v.audio = rel
            v.error = None
            if status and "meter" in status.lower():
                parts = status.split(":", 1)
                if len(parts) == 2:
                    v.meter = parts[1].strip()
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
    pack.meta["audio_resumed"] = skipped
    pack.meta["audio_failed"] = failed
    write_html_pack(pack, root)
    # also dump manifest-friendly annotations already in pack
    log(f"audio: new={ok} resumed={skipped} failed={failed} → {root}")
    return pack


def build_and_synth_chapter(
    chapter: str | int,
    out_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    **kwargs,
) -> LearningPack:
    data = load_sbg_data(data_dir or DEFAULT_DATA_DIR)
    pack = build_chapter_pack(data, chapter)
    return synthesize_pack(pack, out_dir, **kwargs)


def build_all_chapters(
    out_root: str | Path,
    *,
    chapters: list[str] | None = None,
    data_dir: str | Path | None = None,
    publish_docs: bool = False,
    docs_dir: str | Path | None = None,
    **kwargs,
) -> list[Path]:
    """Build every chapter under out_root/ch-NN/. Optionally publish to docs/."""
    log = kwargs.get("progress") or (lambda m: print(m, flush=True))
    data = load_sbg_data(data_dir or DEFAULT_DATA_DIR)
    if chapters is None:
        chapters = [f"{i:02d}" for i in range(1, 19)]
    else:
        chapters = [f"{int(c):02d}" for c in chapters]

    # Load engine once for all chapters
    engine = kwargs.pop("engine", None)
    skip_audio = kwargs.get("skip_audio", False)
    if engine is None and not skip_audio:
        from sanskrit_reciter.engine import LocalEngine
        from sanskrit_reciter.paths import require_assets

        missing = require_assets()
        if missing:
            raise FileNotFoundError("Local assets missing:\n" + "\n".join(missing))
        device = kwargs.get("device", "auto")
        nfe = kwargs.get("nfe", 32)
        log(f"loading TTS engine once (device={device}, nfe={nfe})…")
        engine = LocalEngine(device=device, nfe=nfe)

    out_root = Path(out_root)
    published: list[Path] = []
    for ch in chapters:
        pack = build_chapter_pack(data, ch)
        dest = out_root / f"ch-{ch}"
        log(f"\n======== Chapter {ch}: {len(pack.verses)} verses → {dest} ========")
        synthesize_pack(pack, dest, engine=engine, **kwargs)
        if publish_docs:
            slug = f"gita-ch-{ch}"
            d = publish_pack(dest, docs_dir=docs_dir, slug=slug, rebuild_catalog=False)
            published.append(d)
            log(f"published → {d}")

    if publish_docs:
        from sanskrit_reciter.learning.publish import default_docs_dir

        docs = Path(docs_dir) if docs_dir else default_docs_dir()
        write_catalog(docs)
        log(f"catalog → {docs / 'index.html'}")
    return published
