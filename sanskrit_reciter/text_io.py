"""Read Sanskrit text files and split into individual shlokas."""

from __future__ import annotations

import re
from pathlib import Path

# Catalogue numbers: २.७.८१० or 2.7.810 (often after the verse daṇḍa)
_VERSE_NUM = re.compile(
    r"(?:^|[\s।॥|])"  # start or after space/daṇḍa
    r"[०-९0-9]+(?:\s*[.\u0964]\s*[०-९0-9]+){1,4}"  # a.b.c…
    r"(?:\s*[।॥|])*"  # trailing daṇḍas
)

# Editorial / variant notes printed after the verse, e.g. (पाठ. माहाकुल)
_PAREN_NOTE = re.compile(r"\([^)]*\)")

# Asterisks mark variant readings in some editions
_STARS = re.compile(r"\*+")

# Only digits, daṇḍas, dots, whitespace, Latin punctuation
_NON_TEXT = re.compile(r"^[\s०-९0-9.।॥|,\-–—:;*'\"()\[\]a-zA-Z]+$")

# Minimum akṣaras to treat as a real verse (filters pure numbers / debris)
MIN_AKSHARAS = 6


def read_text(path: str | Path) -> str:
    """Load a UTF-8 text file and normalize line endings."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Text file not found: {p}")
    return p.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def clean_verse(text: str) -> str:
    """Strip edition markup: verse numbers, (पाठ…) notes, asterisks."""
    if not text:
        return ""
    s = text.replace("।।", "॥")
    s = _PAREN_NOTE.sub(" ", s)
    s = _STARS.sub("", s)
    s = _VERSE_NUM.sub(" ", s)
    # Collapse leftover daṇḍa/punct runs and whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip(" \t\n\r|.,;:")
    # Prefer a single closing daṇḍa if the line still has verse body
    if s and not re.search(r"[।॥]$", s) and count_aksharas(s) >= MIN_AKSHARAS:
        # keep as-is; daṇḍa optional for the model
        pass
    return s.strip()


def is_real_verse(text: str) -> bool:
    """True if the chunk looks like Sanskrit worth synthesizing."""
    s = (text or "").strip()
    if not s:
        return False
    if _NON_TEXT.match(s):
        return False
    return count_aksharas(s) >= MIN_AKSHARAS


def split_shlokas(text: str) -> list[str]:
    """Split a multi-verse file into one-shloka chunks.

    Handles common corpus formats:
      • one verse per line with trailing catalogue number
        ``सन्ततिर्… ॥ २.७.८१० ॥``
      • blank-line separated verses
      • classic multi-line ślokas closed by ``॥``

    Catalogue numbers, asterisks, and ``(पाठ…)`` notes are stripped.
    Pure number / empty debris is dropped.
    """
    text = (text or "").strip()
    if not text:
        return []

    normalized = text.replace("।।", "॥")
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]

    # --- Prefer line-oriented split when the file is a catalogue (most lines
    # carry a verse number, or each line is a short complete unit). ----------
    if _looks_like_catalogue(lines):
        out: list[str] = []
        for ln in lines:
            v = clean_verse(ln)
            if is_real_verse(v):
                out.append(v)
        if out:
            return out

    # --- Double-daṇḍa split (classic multi-line ślokas) ---------------------
    if "॥" in normalized:
        parts = re.split(r"(?<=॥)\s*", normalized)
        out = []
        for p in parts:
            v = clean_verse(p)
            if is_real_verse(v):
                out.append(v)
        if out:
            return out

    # --- Blank-line separated -----------------------------------------------
    blanks = re.split(r"\n\s*\n+", normalized)
    out = []
    for p in blanks:
        v = clean_verse(p)
        if is_real_verse(v):
            out.append(v)
    if len(out) > 1:
        return out

    v = clean_verse(normalized)
    return [v] if is_real_verse(v) else []


def _looks_like_catalogue(lines: list[str]) -> bool:
    """Heuristic: Amarakośa-style lines end with a verse number after ॥."""
    if len(lines) < 3:
        return False
    numbered = 0
    for ln in lines[:40]:
        # "… ॥ २.७.८१०" or "… ॥ २.७.८१० ॥"
        if re.search(r"॥\s*[०-९0-9]+(?:\s*[.\u0964]\s*[०-९0-9]+)+", ln):
            numbered += 1
        elif re.search(r"[०-९0-9]+(?:\.[०-९0-9]+){1,3}\s*॥?\s*$", ln):
            numbered += 1
    return numbered >= max(2, len(lines[:40]) // 3)


def count_aksharas(s: str) -> int:
    """Rough akṣara count for Devanagari / Kannada (independent vowels + non-virāma consonants)."""
    n = 0
    L = len(s)
    for i, c in enumerate(s):
        o = ord(c)
        indep = (0x0905 <= o <= 0x0914) or (0x0C85 <= o <= 0x0C94)
        cons = (0x0915 <= o <= 0x0939) or (0x0C95 <= o <= 0x0CB9)
        if indep:
            n += 1
        elif cons:
            nxt = s[i + 1] if i + 1 < L else ""
            if nxt not in ("्", "್"):
                n += 1
    return n
