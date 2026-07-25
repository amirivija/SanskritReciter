"""Publish learning packs into docs/ for GitHub Pages.

Layout::

    docs/
      index.html           # catalog of all packs
      .nojekyll
      <slug>/
        index.html
        manifest.json
        audio/*.wav
        assets/
"""

from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def default_docs_dir(repo_root: Path | None = None) -> Path:
    if repo_root is None:
        # sanskrit_reciter/learning/publish.py → repo root
        repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "docs"


def slugify(name: str) -> str:
    """URL-safe slug from a pack folder name or title."""
    s = name.strip().lower()
    s = s.replace("_", "-").replace(" ", "-")
    # drop learn- prefix noise
    s = re.sub(r"^learn-+", "", s)
    # keep unicode letters/digits; collapse other runs to hyphen
    s = re.sub(r"[^\w\-]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "pack"


@dataclass
class PackCard:
    slug: str
    title: str
    subtitle: str | None
    description: str | None
    verse_count: int
    audio_count: int
    path: str  # relative URL path, e.g. brahma-varga/

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "subtitle": self.subtitle,
            "description": self.description,
            "verse_count": self.verse_count,
            "audio_count": self.audio_count,
            "path": self.path,
        }


def _read_manifest(pack_dir: Path) -> dict[str, Any]:
    m = pack_dir / "manifest.json"
    if not m.is_file():
        raise FileNotFoundError(
            f"Not a learning pack (missing manifest.json): {pack_dir}"
        )
    return json.loads(m.read_text(encoding="utf-8"))


def publish_pack(
    pack_dir: str | Path,
    *,
    docs_dir: str | Path | None = None,
    slug: str | None = None,
    rebuild_catalog: bool = True,
) -> Path:
    """Copy a learning pack into docs/<slug>/ and refresh the catalog.

    Returns the published pack directory path.
    """
    src = Path(pack_dir).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Pack directory not found: {src}")
    manifest = _read_manifest(src)
    if not (src / "index.html").is_file():
        raise FileNotFoundError(f"Pack missing index.html: {src}")

    docs = Path(docs_dir) if docs_dir else default_docs_dir()
    docs.mkdir(parents=True, exist_ok=True)
    (docs / ".nojekyll").write_text("", encoding="utf-8")

    pack_slug = slugify(slug or src.name)
    dest = docs / pack_slug
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"),
    )

    # Ensure relative audio links still work (they already are relative)
    if rebuild_catalog:
        write_catalog(docs)

    return dest


def discover_packs(docs_dir: str | Path) -> list[PackCard]:
    docs = Path(docs_dir)
    if not docs.is_dir():
        return []
    cards: list[PackCard] = []
    for child in sorted(docs.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        man = child / "manifest.json"
        if not man.is_file() or not (child / "index.html").is_file():
            continue
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        verses = data.get("verses") or []
        audio_n = sum(1 for v in verses if v.get("audio"))
        cards.append(
            PackCard(
                slug=child.name,
                title=str(data.get("title") or child.name),
                subtitle=data.get("subtitle"),
                description=data.get("description"),
                verse_count=len(verses),
                audio_count=audio_n,
                path=f"{child.name}/",
            )
        )
    return cards


def write_catalog(docs_dir: str | Path) -> Path:
    """Write docs/index.html listing all published packs."""
    docs = Path(docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / ".nojekyll").write_text("", encoding="utf-8")

    cards = discover_packs(docs)
    catalog_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packs": [c.to_dict() for c in cards],
    }
    (docs / "catalog.json").write_text(
        json.dumps(catalog_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    index = docs / "index.html"
    index.write_text(_render_catalog_html(cards), encoding="utf-8")
    return index


def _esc(s: str | None) -> str:
    return html.escape(s or "", quote=True)


def _render_catalog_html(cards: list[PackCard]) -> str:
    if cards:
        items = "\n".join(_render_card(c) for c in cards)
        grid = f'<div class="grid">{items}</div>'
    else:
        grid = """
        <div class="empty">
          <p>No learning packs published yet.</p>
          <p class="hint">Generate a pack, then publish it:</p>
          <pre>./sr learn examples/learn_demo.txt -o out/learn_demo --title "Demo"
./sr publish out/learn_demo --slug demo</pre>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SanskritReciter · Learning packs</title>
  <style>
{_CATALOG_CSS}
  </style>
</head>
<body>
  <header class="hero">
    <p class="eyebrow">SanskritReciter</p>
    <h1>Learning packs</h1>
    <p class="lede">
      Śloka text with chanted audio — study offline-friendly packs hosted on GitHub Pages.
    </p>
  </header>
  <main>
    {grid}
  </main>
  <footer>
    <p>Built with <a href="https://github.com/prathoshap/vagdhenu">Vāgdhenu</a> ·
       open a pack to play each śloka in place.</p>
  </footer>
</body>
</html>
"""


def _render_card(c: PackCard) -> str:
    sub = f'<p class="sub">{_esc(c.subtitle)}</p>' if c.subtitle else ""
    desc = f'<p class="desc">{_esc(c.description)}</p>' if c.description else ""
    return f"""
    <a class="card" href="{_esc(c.path)}">
      <h2>{_esc(c.title)}</h2>
      {sub}
      {desc}
      <p class="meta">{c.verse_count} śloka(s) · {c.audio_count} with audio</p>
      <span class="go">Open pack →</span>
    </a>"""


_CATALOG_CSS = """
  :root {
    --bg: #f7f3ea;
    --card: #fffdf7;
    --ink: #1c1917;
    --muted: #78716c;
    --accent: #9a3412;
    --line: #e7e0d2;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
    color: var(--ink);
    background:
      radial-gradient(900px 480px at 0% -10%, #fff8e7, transparent),
      radial-gradient(800px 400px at 100% 0%, #f0e6d4, transparent),
      var(--bg);
    line-height: 1.55;
    min-height: 100vh;
  }
  .hero, main, footer {
    width: min(900px, calc(100% - 2rem));
    margin-left: auto; margin-right: auto;
  }
  .hero { padding: 2.5rem 0 1.2rem; }
  .eyebrow {
    text-transform: uppercase; letter-spacing: 0.12em;
    font-size: 0.78rem; color: var(--accent); margin: 0 0 0.4rem; font-weight: 700;
  }
  h1 {
    font-family: "Noto Serif Devanagari", "Georgia", serif;
    font-size: clamp(1.8rem, 4vw, 2.4rem); margin: 0 0 0.5rem; color: var(--accent);
  }
  .lede { color: var(--muted); max-width: 52ch; margin: 0; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1rem;
    padding: 0.5rem 0 2.5rem;
  }
  .card {
    display: block;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.15rem 1.2rem;
    text-decoration: none;
    color: inherit;
    box-shadow: 0 8px 24px rgba(68, 42, 12, 0.04);
    transition: border-color 0.15s, transform 0.15s;
  }
  .card:hover {
    border-color: #d6a57a;
    transform: translateY(-2px);
  }
  .card h2 {
    font-family: "Noto Serif Devanagari", "Georgia", serif;
    font-size: 1.25rem; margin: 0 0 0.35rem; color: var(--accent);
  }
  .sub, .desc, .meta { margin: 0.25rem 0; color: var(--muted); font-size: 0.95rem; }
  .desc { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
  .go { display: inline-block; margin-top: 0.7rem; color: var(--accent); font-weight: 600; font-size: 0.92rem; }
  .empty {
    background: var(--card); border: 1px dashed var(--line); border-radius: 14px;
    padding: 1.5rem; color: var(--muted);
  }
  .empty pre {
    background: #f3eee4; padding: 0.8rem 1rem; border-radius: 8px;
    overflow-x: auto; color: var(--ink); font-size: 0.85rem;
  }
  footer { padding: 0 0 2.5rem; color: var(--muted); font-size: 0.88rem; }
  footer a { color: var(--accent); }
"""
