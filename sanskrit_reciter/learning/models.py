"""Extensible data model for a Sanskrit learning pack.

A pack is a sequence of verses. Each verse has:
  - source text (śloka)
  - optional recitation audio
  - zero or more annotations (anvaya, pratipadārtha, grammar, …)

New literary decorations are added by registering annotation *kinds* and
attaching Annotation objects — the HTML exporter picks them up without
schema churn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AnnotationKind(str, Enum):
    """Well-known annotation kinds. Custom kinds use free-form strings."""

    ANVAYA = "anvaya"  # prose word-order reconstruction
    PRATIPADA = "pratipada"  # word-by-word meaning
    GRAMMAR = "grammar"  # vyākaraṇa notes
    LITERARY = "literary"  # alankāra, rasa, metre notes
    TRANSLATION = "translation"  # free translation
    NOTES = "notes"  # free-form teacher notes
    CUSTOM = "custom"


# Display labels (Devanagari + English) for known kinds — used in HTML.
ANNOTATION_LABELS: dict[str, str] = {
    AnnotationKind.ANVAYA.value: "अन्वय · Anvaya",
    AnnotationKind.PRATIPADA.value: "प्रतिपदार्य · Word meanings",
    AnnotationKind.GRAMMAR.value: "व्याकरण · Grammar",
    AnnotationKind.LITERARY.value: "साहित्य · Literary notes",
    AnnotationKind.TRANSLATION.value: "अर्थ · Translation",
    AnnotationKind.NOTES.value: "टिप्पणी · Notes",
}


@dataclass
class Annotation:
    """One pedagogical decoration attached to a verse.

    ``content`` is intentionally flexible:
      - str  → prose / markdown-ish plain text
      - list[str] → bullet list
      - list[dict] → structured rows (e.g. pratipada: [{pada, artha}, …])
      - dict → free structured payload

    ``render`` hints how the HTML layer should present it:
      auto | prose | list | table | hidden
    """

    kind: str
    content: Any
    title: str | None = None
    render: str = "auto"
    meta: dict[str, Any] = field(default_factory=dict)

    def display_title(self) -> str:
        if self.title:
            return self.title
        return ANNOTATION_LABELS.get(self.kind, self.kind.replace("_", " ").title())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "render": self.render,
            "content": self.content,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Annotation:
        return cls(
            kind=str(d.get("kind") or AnnotationKind.CUSTOM.value),
            content=d.get("content"),
            title=d.get("title"),
            render=str(d.get("render") or "auto"),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class VerseUnit:
    """One śloka (or half-verse unit) in a learning pack."""

    id: str  # stable id, e.g. "001"
    text: str  # Sanskrit source
    index: int = 0  # 1-based display order
    audio: str | None = None  # path relative to pack root, e.g. audio/001.wav
    meter: str | None = None
    source_ref: str | None = None  # e.g. "२.७.८१०" if known
    annotations: list[Annotation] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None  # set if synthesis failed

    def annotation(self, kind: str) -> Annotation | None:
        for a in self.annotations:
            if a.kind == kind:
                return a
        return None

    def set_annotation(self, ann: Annotation) -> None:
        """Replace existing annotation of the same kind, or append."""
        for i, a in enumerate(self.annotations):
            if a.kind == ann.kind:
                self.annotations[i] = ann
                return
        self.annotations.append(ann)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "text": self.text,
            "audio": self.audio,
            "meter": self.meter,
            "source_ref": self.source_ref,
            "annotations": [a.to_dict() for a in self.annotations],
            "meta": self.meta,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerseUnit:
        return cls(
            id=str(d["id"]),
            text=str(d.get("text") or ""),
            index=int(d.get("index") or 0),
            audio=d.get("audio"),
            meter=d.get("meter"),
            source_ref=d.get("source_ref"),
            annotations=[Annotation.from_dict(a) for a in (d.get("annotations") or [])],
            meta=dict(d.get("meta") or {}),
            error=d.get("error"),
        )


@dataclass
class LearningPack:
    """A complete learning product unit: title + ordered verses + assets."""

    title: str
    verses: list[VerseUnit] = field(default_factory=list)
    subtitle: str | None = None
    description: str | None = None
    language: str = "sa"  # primary script language tag
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "description": self.description,
            "language": self.language,
            "meta": self.meta,
            "verses": [v.to_dict() for v in self.verses],
            "schema_version": 1,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearningPack:
        return cls(
            title=str(d.get("title") or "Sanskrit Learning Pack"),
            subtitle=d.get("subtitle"),
            description=d.get("description"),
            language=str(d.get("language") or "sa"),
            meta=dict(d.get("meta") or {}),
            verses=[VerseUnit.from_dict(v) for v in (d.get("verses") or [])],
        )

    def verse_by_id(self, vid: str) -> VerseUnit | None:
        for v in self.verses:
            if v.id == vid:
                return v
        return None
