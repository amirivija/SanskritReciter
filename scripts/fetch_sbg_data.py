#!/usr/bin/env python3
"""Download Bhagavad Gītā e-reader JSON from samsaadhanii/scl into data/sbg/."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sbg"
BASE = (
    "https://raw.githubusercontent.com/samsaadhanii/scl/master/"
    "e-readers/SBG-NEW/sbg_ereader/assets/data"
)
FILES = [
    "sloka.json",
    "analysis.json",
    "chapters.json",
    "about.json",
    "intro.json",
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"{BASE}/{name}"
        dest = OUT / name
        print(f"↓ {name} …", flush=True)
        urllib.request.urlretrieve(url, dest)
        print(f"  → {dest} ({dest.stat().st_size:,} bytes)")
    print(f"✓ SBG data in {OUT}")
    print("  Source: https://github.com/samsaadhanii/scl")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"failed: {e}", file=sys.stderr)
        raise
