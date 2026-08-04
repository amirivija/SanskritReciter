"""CLI: recite audio + learning packs (fully local)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from sanskrit_reciter import __version__
from sanskrit_reciter.paths import pick_device, require_assets
from sanskrit_reciter.text_io import count_aksharas, read_text, split_shlokas

DEFAULT_VERSE_GAP_S = 0.75


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sanskrit-reciter",
        description=(
            "Sanskrit chant TTS and learning packs (local Vāgdhenu pipeline)."
        ),
    )
    p.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = p.add_subparsers(dest="command")

    # ── recite ────────────────────────────────────────────────────────────
    r = sub.add_parser(
        "recite",
        help="Synthesize chant audio from a text file",
        description="Text file → WAV (single verse or concatenated --all).",
    )
    _add_recite_args(r)
    r.set_defaults(func=cmd_recite)

    # ── learn ─────────────────────────────────────────────────────────────
    l = sub.add_parser(
        "learn",
        help="Build an HTML learning pack (śloka + audio + annotation slots)",
        description=(
            "Generate a self-contained learning folder: each śloka with its "
            "recitation audio, ready for anvaya / pratipadārtha / grammar later."
        ),
    )
    _add_learn_args(l)
    l.set_defaults(func=cmd_learn)

    # ── publish (GitHub Pages via docs/) ──────────────────────────────────
    p_pub = sub.add_parser(
        "publish",
        help="Copy a learning pack into docs/ for GitHub Pages",
        description=(
            "Publish a pack built by `learn` into docs/<slug>/ and refresh "
            "the catalog at docs/index.html. Enable Pages: Settings → Pages → "
            "Deploy from branch → /docs (or use the included Actions workflow)."
        ),
    )
    p_pub.add_argument(
        "pack_dir",
        nargs="?",
        default=None,
        help="Path to a learning pack directory (contains index.html + manifest.json)",
    )
    p_pub.add_argument(
        "--slug",
        default=None,
        help="URL slug under docs/ (default: derived from pack folder name)",
    )
    p_pub.add_argument(
        "--docs-dir",
        default=None,
        help="Docs root (default: <repo>/docs)",
    )
    p_pub.add_argument(
        "--catalog-only",
        action="store_true",
        help="Only rebuild docs/index.html from packs already under docs/",
    )
    p_pub.add_argument(
        "--list",
        action="store_true",
        dest="list_packs",
        help="List packs currently under docs/",
    )
    p_pub.set_defaults(func=cmd_publish)

    # ── gita (Bhagavad Gītā from Samsaadhanii SCL data) ───────────────────
    g = sub.add_parser(
        "gita",
        help="Build Bhagavad Gītā learning packs (text + grammar + audio)",
        description=(
            "Import ślokas and grammatical analysis from Samsaadhanii SCL "
            "(https://github.com/samsaadhanii/scl), synthesize recitation, "
            "and optionally publish chapter packs to docs/ for GitHub Pages."
        ),
    )
    g.add_argument(
        "--chapter",
        "-c",
        action="append",
        dest="chapters",
        help="Chapter number 1–18 (repeatable). Default: all 18 chapters",
    )
    g.add_argument(
        "-o",
        "--output",
        default="out/gita",
        help="Output root directory (default: out/gita → ch-01 … ch-18)",
    )
    g.add_argument(
        "--data-dir",
        default=None,
        help="Path to SBG JSON data (default: data/sbg)",
    )
    g.add_argument(
        "--publish",
        action="store_true",
        help="Copy finished chapter packs into docs/gita-ch-NN/ and refresh catalog",
    )
    g.add_argument(
        "--skip-audio",
        action="store_true",
        help="HTML + grammar only (no TTS)",
    )
    g.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-synthesize even if audio files already exist",
    )
    g.add_argument(
        "--fetch",
        action="store_true",
        help="Download/refresh SBG JSON from GitHub before building",
    )
    _add_common_synth(g)
    # Most of the Gītā is anuṣṭubh; auto-detect often fails on sandhi-split text.
    g.set_defaults(func=cmd_gita, meter="anuṣṭubh")

    # ── list / check (top-level convenience) ──────────────────────────────
    c = sub.add_parser("check", help="Verify local model assets")
    c.add_argument("--device", default="auto")
    c.set_defaults(func=cmd_check)

    return p


def _add_common_synth(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--device",
        default="auto",
        help="torch device: auto|mps|cpu|cuda (default: auto)",
    )
    p.add_argument(
        "--meter",
        default="__auto__",
        help="Chandas / meter (default: auto-detect)",
    )
    p.add_argument("--seed", type=int, default=60, help="Random seed (default: 60)")
    p.add_argument(
        "--nfe",
        type=int,
        default=32,
        help="ODE steps (default 32; 64 closer to production)",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip verses that fail and keep going",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Stop on first verse failure (overrides continue-on-error)",
    )


def _add_recite_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("text_file", help="UTF-8 text file of Sanskrit verse(s)")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output WAV (default: <text_file stem>.wav). "
        "With --all, all verses are concatenated into this file.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Render every shloka into one continuous audio file",
    )
    p.add_argument(
        "--gap",
        type=float,
        default=DEFAULT_VERSE_GAP_S,
        help=f"Silence between verses with --all (default: {DEFAULT_VERSE_GAP_S}s)",
    )
    p.add_argument(
        "--save-parts",
        action="store_true",
        help="With --all, also write per-verse WAVs",
    )
    p.add_argument(
        "--list-shlokas",
        action="store_true",
        help="Parse and print shlokas without synthesizing",
    )
    _add_common_synth(p)


def _add_learn_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("text_file", help="UTF-8 text file of Sanskrit verse(s)")
    p.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for the learning pack (index.html + audio/)",
    )
    p.add_argument("--title", default=None, help="Pack title (default: from filename)")
    p.add_argument("--subtitle", default=None, help="Optional subtitle")
    p.add_argument("--description", default=None, help="Optional pack description")
    p.add_argument(
        "--annotations",
        default=None,
        help="Optional JSON sidecar with anvaya / pratipada / grammar / …",
    )
    p.add_argument(
        "--skip-audio",
        action="store_true",
        help="Build HTML + manifest only (no TTS); useful with existing audio or dry-run",
    )
    p.add_argument(
        "--list-shlokas",
        action="store_true",
        help="Parse and print shlokas without building the pack",
    )
    _add_common_synth(p)


def _resolve_continue_on_error(args: argparse.Namespace, *, default: bool) -> bool:
    """learn defaults to continue; recite defaults to stop. --strict always stops."""
    if getattr(args, "strict", False):
        return False
    if getattr(args, "continue_on_error", False):
        return True
    return default


def cmd_check(args: argparse.Namespace) -> int:
    device = pick_device(args.device)
    missing = require_assets()
    print(f"device: {device}")
    if missing:
        print("assets: MISSING")
        print("\n".join(missing))
        print("\nRun: python scripts/download_models.py")
        return 1
    print("assets: OK (all weights + vendor code present)")
    return 0


def cmd_recite(args: argparse.Namespace) -> int:
    try:
        raw = read_text(args.text_file)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    shlokas = split_shlokas(raw)
    if not shlokas:
        print("error: no Sanskrit text found in file", file=sys.stderr)
        return 1

    if args.list_shlokas:
        for i, s in enumerate(shlokas, 1):
            n = count_aksharas(s)
            preview = s.replace("\n", " ")[:80]
            print(f"[{i}] ~{n} akṣaras  {preview}{'…' if len(s) > 80 else ''}")
        print(f"\n{len(shlokas)} shloka(s)")
        return 0

    missing = require_assets()
    if missing:
        print("error: local assets missing\n" + "\n".join(missing), file=sys.stderr)
        return 1

    to_render = shlokas if args.all else shlokas[:1]
    if not args.all and len(shlokas) > 1:
        print(
            f"note: file has {len(shlokas)} shlokas; rendering the first only "
            f"(pass --all for one continuous file, or use: learn …)",
            file=sys.stderr,
        )

    from sanskrit_reciter.engine import LocalEngine

    try:
        engine = LocalEngine(device=args.device, nfe=args.nfe)
    except Exception as e:
        print(f"error: failed to load models: {e}", file=sys.stderr)
        return 1

    base_out = (
        Path(args.output)
        if args.output
        else Path(args.text_file).with_suffix(".wav")
    )
    base_out.parent.mkdir(parents=True, exist_ok=True)

    multi = len(to_render) > 1
    print(
        f"device: {engine.device}  meter: {args.meter}  seed: {args.seed}  "
        f"nfe: {args.nfe}" + (f"  gap: {args.gap}s" if multi else "")
    )
    print(
        f"rendering {len(to_render)} verse(s)"
        + (" → single continuous audio" if multi else "")
        + "…"
    )

    ok = failed = 0
    segments: list[np.ndarray] = []
    sample_rate: int | None = None
    width = max(2, len(str(len(to_render))))

    for i, text in enumerate(to_render, 1):
        label = f"[{i}/{len(to_render)}]"
        preview = text.replace("\n", " ")[:60]
        print(f"{label} synthesizing…  {preview}{'…' if len(text) > 60 else ''}")
        try:
            sr, audio, status = engine.synthesize(
                text, meter=args.meter, seed=args.seed
            )
        except Exception as e:
            failed += 1
            print(f"{label} failed: {e}", file=sys.stderr)
            if _resolve_continue_on_error(args, default=False):
                continue
            print(
                "(tip: pass --continue-on-error to skip failures)",
                file=sys.stderr,
            )
            return 1

        if sample_rate is None:
            sample_rate = int(sr)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        segments.append(audio)
        ok += 1
        print(f"{label} {status or 'ok'}  ({len(audio) / sr:.1f}s)")

        if args.save_parts and multi:
            part_path = base_out.with_name(
                f"{base_out.stem}_{i:0{width}d}{base_out.suffix or '.wav'}"
            )
            sf.write(str(part_path), audio, int(sr))
            print(f"{label} part → {part_path}")

    if not segments or sample_rate is None:
        print("error: no verses rendered successfully", file=sys.stderr)
        return 1

    combined = _stitch(segments, args.gap if multi else 0.0, sample_rate)
    peak = float(np.abs(combined).max()) if combined.size else 0.0
    if peak > 1.0:
        combined = combined / peak * 0.97

    out_path = base_out if base_out.suffix else base_out.with_suffix(".wav")
    sf.write(str(out_path), combined, sample_rate)
    dur = len(combined) / sample_rate
    print(
        f"wrote {out_path}  ({out_path.stat().st_size} bytes, {dur:.1f}s, "
        f"{ok} verse(s)" + (f", {failed} skipped" if failed else "") + ")"
    )
    if multi:
        print(f"done: {ok} ok, {failed} failed, {len(to_render)} total")
    return 0 if ok else 1


def cmd_learn(args: argparse.Namespace) -> int:
    try:
        raw = read_text(args.text_file)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    shlokas = split_shlokas(raw)
    if not shlokas:
        print("error: no Sanskrit text found in file", file=sys.stderr)
        return 1

    if args.list_shlokas:
        for i, s in enumerate(shlokas, 1):
            n = count_aksharas(s)
            preview = s.replace("\n", " ")[:80]
            print(f"[{i}] ~{n} akṣaras  {preview}{'…' if len(s) > 80 else ''}")
        print(f"\n{len(shlokas)} shloka(s)")
        return 0

    # learn is corpus-oriented: continue-on-error defaults to True (--strict to stop)
    continue_on_error = _resolve_continue_on_error(args, default=True)

    if not args.skip_audio:
        missing = require_assets()
        if missing:
            print("error: local assets missing\n" + "\n".join(missing), file=sys.stderr)
            return 1

    from sanskrit_reciter.learning.builder import build_learning_pack

    try:
        pack, index = build_learning_pack(
            args.text_file,
            args.output,
            title=args.title,
            subtitle=args.subtitle,
            description=args.description,
            annotations_file=args.annotations,
            device=args.device,
            nfe=args.nfe,
            meter=args.meter,
            seed=args.seed,
            continue_on_error=continue_on_error,
            skip_audio=args.skip_audio,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    n_audio = sum(1 for v in pack.verses if v.audio)
    n_fail = sum(1 for v in pack.verses if v.error)
    print(
        f"learning pack ready: {index}\n"
        f"  verses={len(pack.verses)}  audio={n_audio}  failed={n_fail}\n"
        f"  open the HTML file in a browser to study with audio"
    )
    return 0 if n_audio or args.skip_audio else 1


def cmd_gita(args: argparse.Namespace) -> int:
    from sanskrit_reciter.learning.sbg import DEFAULT_DATA_DIR, load_sbg_data
    from sanskrit_reciter.learning.sbg_builder import build_all_chapters

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    if args.fetch:
        import urllib.request

        data_dir.mkdir(parents=True, exist_ok=True)
        base = (
            "https://raw.githubusercontent.com/samsaadhanii/scl/master/"
            "e-readers/SBG-NEW/sbg_ereader/assets/data"
        )
        for name in (
            "sloka.json",
            "analysis.json",
            "chapters.json",
            "about.json",
            "intro.json",
        ):
            print(f"↓ {name}")
            urllib.request.urlretrieve(f"{base}/{name}", data_dir / name)
        print(f"✓ data → {data_dir}")
    try:
        data = load_sbg_data(data_dir)
    except FileNotFoundError as e:
        print(f"error: {e}\nTip: ./sr gita --fetch …", file=sys.stderr)
        return 1

    chapters = None
    if args.chapters:
        chapters = [str(c) for c in args.chapters]

    n_sloka = len({(s.get("chaptno"), s.get("slokano")) for s in data["sloka"]})
    print(
        f"SBG data: {n_sloka} unique ślokas, "
        f"{len(data['analysis'])} analysis rows"
    )
    print(
        f"building chapters={chapters or 'all 1–18'} → {args.output}"
        + (" (+ publish to docs/)" if args.publish else "")
    )

    continue_on_error = _resolve_continue_on_error(args, default=True)
    try:
        build_all_chapters(
            args.output,
            chapters=chapters,
            data_dir=data_dir,
            publish_docs=args.publish,
            device=args.device,
            nfe=args.nfe,
            meter=args.meter,
            seed=args.seed,
            continue_on_error=continue_on_error,
            resume=not args.no_resume,
            skip_audio=args.skip_audio,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(
        f"done. packs under {args.output}/ch-NN/\n"
        + (
            "published under docs/gita-ch-NN/ — commit & push to update GitHub Pages"
            if args.publish
            else "use: ./sr gita … --publish   to copy into docs/"
        )
    )
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    from sanskrit_reciter.learning.publish import (
        default_docs_dir,
        discover_packs,
        publish_pack,
        write_catalog,
    )

    docs = Path(args.docs_dir) if args.docs_dir else default_docs_dir()

    if args.list_packs:
        cards = discover_packs(docs)
        if not cards:
            print(f"no packs under {docs}")
            return 0
        for c in cards:
            print(f"  {c.slug:24}  {c.verse_count:4} verses  {c.title}")
        print(f"\n{len(cards)} pack(s) in {docs}")
        return 0

    if args.catalog_only:
        index = write_catalog(docs)
        cards = discover_packs(docs)
        print(f"catalog rebuilt: {index}  ({len(cards)} pack(s))")
        return 0

    if not args.pack_dir:
        print(
            "error: pack_dir required (or use --catalog-only / --list)\n"
            "  ./sr publish out/learn_demo --slug demo",
            file=sys.stderr,
        )
        return 2

    try:
        dest = publish_pack(
            args.pack_dir,
            docs_dir=docs,
            slug=args.slug,
            rebuild_catalog=True,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    cards = discover_packs(docs)
    print(
        f"published → {dest}\n"
        f"catalog   → {docs / 'index.html'}  ({len(cards)} pack(s))\n"
        f"\nNext: commit docs/ and enable GitHub Pages (Deploy from branch → /docs),\n"
        f"or push and use the pages.yml workflow. Site path: …/docs/{dest.name}/"
    )
    return 0


def _stitch(segments: list[np.ndarray], gap_s: float, sr: int) -> np.ndarray:
    if not segments:
        return np.zeros(0, dtype=np.float32)
    if len(segments) == 1:
        return np.asarray(segments[0], dtype=np.float32)
    gap = np.zeros(max(0, int(gap_s * sr)), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        parts.append(np.asarray(seg, dtype=np.float32).reshape(-1))
        if i < len(segments) - 1 and gap.size:
            parts.append(gap)
    return np.concatenate(parts)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    # Backward compatible: bare `sanskrit-reciter FILE …` → recite
    if argv and not argv[0].startswith("-") and argv[0] not in (
        "recite",
        "learn",
        "publish",
        "gita",
        "check",
        "help",
    ):
        argv = ["recite"] + argv

    if not argv:
        parser.print_help()
        print(
            "\nCommands:\n"
            "  recite   TEXT  -o audio.wav [--all]\n"
            "  learn    TEXT  -o pack_dir/     # HTML + per-śloka audio\n"
            "  gita     [--chapter N] [--publish]  # full Bhagavad Gītā packs\n"
            "  publish  PACK  [--slug NAME]    # copy pack → docs/ (GitHub Pages)\n"
            "  check\n",
            file=sys.stderr,
        )
        return 2

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
