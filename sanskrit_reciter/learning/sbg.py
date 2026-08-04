"""Import Bhagavad Gītā data from Samsaadhanii SCL e-reader assets.

Source (attribution required)::

    https://github.com/samsaadhanii/scl
    e-readers/SBG-NEW/sbg_ereader/assets/data/

Files used:
  - sloka.json     — verse text (spart1 / spart2)
  - analysis.json  — word-level morphology, kāraka, meanings, anvaya order
  - chapters.json  — chapter titles
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from sanskrit_reciter.learning.models import Annotation, LearningPack, VerseUnit

# Default local cache of the upstream JSON (downloaded by scripts/fetch_sbg_data.py)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sbg"

SOURCE_URL = (
    "https://github.com/samsaadhanii/scl/tree/master/"
    "e-readers/SBG-NEW/sbg_ereader/assets/data"
)
SOURCE_CREDIT = (
    "Text & grammatical analysis from Samsaadhanii SCL "
    f"(Śrīmad Bhagavad Gītā e-reader) — {SOURCE_URL}"
)


def _zch(x: Any) -> str:
    return str(x).strip().zfill(2)


def _zsl(x: Any) -> str:
    return str(x).strip().zfill(3)


def _anvaya_key(row: dict[str, Any]) -> tuple:
    raw = str(row.get("anvaya_no") or "0")
    parts = raw.split(".")
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            nums.append(0.0)
    while len(nums) < 3:
        nums.append(0.0)
    return tuple(nums)


def _clean_display_text(spart1: str, spart2: str) -> str:
    """Join hemistichs; drop editorial hyphens used for sandhi display."""
    a = re.sub(r"\s*-\s*", "", (spart1 or "").strip())
    b = re.sub(r"\s*-\s*", "", (spart2 or "").strip())
    # normalize spaces
    a = re.sub(r"\s+", " ", a)
    b = re.sub(r"\s+", " ", b)
    if a and b:
        return f"{a}\n{b}"
    return a or b


def load_sbg_data(data_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    required = ["sloka.json", "analysis.json", "chapters.json"]
    missing = [f for f in required if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"SBG data missing under {root}: {', '.join(missing)}\n"
            "Run: python scripts/fetch_sbg_data.py"
        )
    return {
        "root": root,
        "sloka": json.loads((root / "sloka.json").read_text(encoding="utf-8")),
        "analysis": json.loads((root / "analysis.json").read_text(encoding="utf-8")),
        "chapters": json.loads((root / "chapters.json").read_text(encoding="utf-8")),
        "about": _load_optional(root / "about.json"),
        "intro": _load_optional(root / "intro.json"),
    }


def _load_optional(path: Path) -> Any:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _index_analysis(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict]]:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        # upstream key is misspelled "chpatno"
        ch = _zch(row.get("chpatno") or row.get("chaptno") or "")
        sl = _zsl(row.get("slokano") or "")
        if ch == "00" or sl == "000":
            continue
        by[(ch, sl)].append(row)
    return by


def _unique_slokas(sloka_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate (chaptno, slokano), keep first occurrence."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for s in sloka_list:
        key = (_zch(s.get("chaptno")), _zsl(s.get("slokano")))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def chapter_title(chapters: list[dict[str, Any]], ch: str) -> str:
    ch = _zch(ch)
    for c in chapters:
        if _zch(c.get("route")) == ch:
            return str(c.get("chapt") or f"अध्यायः {ch}")
    return f"अध्यायः {ch}"


def build_annotations_for_rows(rows: list[dict[str, Any]]) -> list[Annotation]:
    """Turn word-level analysis rows into pack annotations."""
    if not rows:
        return []

    ordered = sorted(rows, key=_anvaya_key)
    anvaya_words = [r.get("word") or "" for r in ordered if r.get("word")]
    anvaya = " ".join(anvaya_words).strip()

    table: list[dict[str, str]] = []
    for r in ordered:
        word = (r.get("word") or "").strip()
        if not word:
            continue
        table.append(
            {
                "pada": word,
                "sandhi": (r.get("sandhied_word") or "-").strip(),
                "morph": (r.get("morph_in_context") or r.get("morph_analysis") or "-").strip(),
                "kaaraka": (r.get("kaaraka_sambandha") or "-").strip(),
                "artha": (r.get("english_meaning") or "-").strip(),
                "hindi": (r.get("hindi_meaning") or "-").strip(),
                "samasa": (r.get("samAsa") or "-").strip(),
            }
        )

    anns: list[Annotation] = []
    if anvaya:
        anns.append(
            Annotation(
                kind="anvaya",
                title="अन्वय · Anvaya",
                content=anvaya,
                render="prose",
                meta={"source": "samsaadhanii/scl"},
            )
        )
    if table:
        anns.append(
            Annotation(
                kind="pratipada",
                title="प्रतिपदार्य · Word meanings",
                content=[{"pada": t["pada"], "artha": t["artha"], "hindi": t["hindi"]} for t in table],
                render="table",
                meta={"source": "samsaadhanii/scl"},
            )
        )
        anns.append(
            Annotation(
                kind="grammar",
                title="व्याकरण · Grammatical analysis",
                content=[
                    {
                        "pada": t["pada"],
                        "morph": t["morph"],
                        "kaaraka": t["kaaraka"],
                        "sandhi": t["sandhi"],
                        "samasa": t["samasa"],
                    }
                    for t in table
                ],
                render="table",
                meta={"source": "samsaadhanii/scl"},
            )
        )
    return anns


def iter_chapter_slokas(
    data: dict[str, Any],
    chapter: str | int | None = None,
) -> Iterable[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    """Yield (chapter_no, sloka_dict, analysis_rows) for one or all chapters."""
    slokas = _unique_slokas(data["sloka"])
    by_analysis = _index_analysis(data["analysis"])
    want = _zch(chapter) if chapter is not None else None
    for s in slokas:
        ch = _zch(s.get("chaptno"))
        if want and ch != want:
            continue
        sl = _zsl(s.get("slokano"))
        yield ch, s, by_analysis.get((ch, sl), [])


def build_chapter_pack(
    data: dict[str, Any],
    chapter: str | int,
) -> LearningPack:
    """Build a LearningPack (no audio) for one adhyāya."""
    ch = _zch(chapter)
    title = chapter_title(data["chapters"], ch)
    pack = LearningPack(
        title=f"भगवद्गीता · {title}",
        subtitle=f"Chapter {int(ch)} · Śrīmad Bhagavad Gītā",
        description=SOURCE_CREDIT,
        language="sa",
        meta={
            "work": "bhagavad_gita",
            "chapter": ch,
            "source": SOURCE_URL,
            "attribution": SOURCE_CREDIT,
        },
    )

    verses: list[VerseUnit] = []
    for i, (ch_no, s, rows) in enumerate(iter_chapter_slokas(data, ch), 1):
        sl = _zsl(s.get("slokano"))
        text = _clean_display_text(s.get("spart1", ""), s.get("spart2", ""))
        vid = f"{ch_no}-{sl}"
        unit = VerseUnit(
            id=vid,
            text=text,
            index=i,
            source_ref=f"{int(ch_no)}.{int(sl)}",
            annotations=build_annotations_for_rows(rows),
            meta={
                "chapter": ch_no,
                "sloka": sl,
                "has_analysis": bool(rows),
                "word_count": len(rows),
            },
        )
        if not rows:
            unit.set_annotation(
                Annotation(
                    kind="notes",
                    title="Notes",
                    content="Grammatical analysis not available for this verse in the SCL dataset.",
                    render="prose",
                )
            )
        verses.append(unit)

    pack.verses = verses
    pack.meta["verse_count"] = len(verses)
    return pack


def build_full_gita_packs(data: dict[str, Any]) -> list[LearningPack]:
    chapters = sorted({_zch(s.get("chaptno")) for s in _unique_slokas(data["sloka"])})
    return [build_chapter_pack(data, ch) for ch in chapters]


def write_chapter_text_file(pack: LearningPack, path: str | Path) -> Path:
    """Write plain text (one verse block, blank-line separated) for tooling."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for v in pack.verses:
        blocks.append(v.text.strip() + " ॥")
    p.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return p


def annotations_sidecar_dict(pack: LearningPack) -> dict[str, Any]:
    """Export annotations in the sidecar format used by --annotations."""
    verses = {}
    for v in pack.verses:
        verses[v.id] = {
            "source_ref": v.source_ref,
            "annotations": [a.to_dict() for a in v.annotations],
            "meta": v.meta,
        }
    return {
        "title": pack.title,
        "subtitle": pack.subtitle,
        "description": pack.description,
        "meta": pack.meta,
        "verses": verses,
    }
