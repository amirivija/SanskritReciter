"""Load optional annotation sidecars and merge into a LearningPack.

Sidecar JSON (optional, for future content authors)::

    {
      "title": "…",            # optional pack overrides
      "verses": {
        "001": {
          "source_ref": "२.७.८१०",
          "annotations": [
            {"kind": "anvaya", "content": "…"},
            {
              "kind": "pratipada",
              "render": "table",
              "content": [
                {"pada": "सन्ततिः", "artha": "lineage"},
                {"pada": "गोत्रम्", "artha": "clan"}
              ]
            },
            {"kind": "grammar", "content": "…"},
            {"kind": "translation", "content": "…"}
          ]
        },
        "002": { … }
      }
    }

Keys may also be 1-based integer strings matching verse index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sanskrit_reciter.learning.models import Annotation, LearningPack


def load_annotations_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Annotations file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def merge_annotations(pack: LearningPack, data: dict[str, Any]) -> LearningPack:
    """Apply sidecar data onto an existing pack (in place) and return it."""
    if data.get("title"):
        pack.title = str(data["title"])
    if data.get("subtitle"):
        pack.subtitle = str(data["subtitle"])
    if data.get("description"):
        pack.description = str(data["description"])
    if isinstance(data.get("meta"), dict):
        pack.meta.update(data["meta"])

    verses_map = data.get("verses") or {}
    if not isinstance(verses_map, dict):
        return pack

    for v in pack.verses:
        entry = verses_map.get(v.id) or verses_map.get(str(v.index)) or verses_map.get(
            f"{v.index:03d}"
        )
        if not entry:
            continue
        if entry.get("source_ref"):
            v.source_ref = str(entry["source_ref"])
        if entry.get("text"):
            # allow curated orthography override
            v.text = str(entry["text"])
        if isinstance(entry.get("meta"), dict):
            v.meta.update(entry["meta"])
        for raw in entry.get("annotations") or []:
            if isinstance(raw, dict):
                v.set_annotation(Annotation.from_dict(raw))
    return pack
