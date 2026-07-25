# SanskritReciter

**Fully local** tool: Sanskrit text file → chanted recitation audio (pārāyaṇa).  
No network calls at runtime. Powered by [Vāgdhenu](https://github.com/prathoshap/vagdhenu) (Prof. Prathosh, IISc).

## Requirements

- Python **3.10+**
- **~4–6 GB** disk for model weights
- Device (auto-selected):
  - **Apple Silicon** → Metal (MPS)
  - **NVIDIA GPU** → CUDA
  - else → CPU (slow)

## One-time setup (needs network once)

```bash
cd SanskritReciter
python3 -m venv .venv && source .venv/bin/activate
bash scripts/setup_local.sh
# or step by step:
#   pip install torch torchaudio
#   pip install -r requirements.txt && pip install -e .
#   python scripts/download_models.py
```

This installs deps, ensures `vendor/vagdhenu` + `vendor/BigVGAN`, and downloads weights into `models/`.

## Generate audio (offline)

**Use the project venv** — system `python3` on macOS is 3.9 and will fail with `No module named 'soundfile'`.

```bash
cd /path/to/SanskritReciter

# Recommended: ./sr always uses .venv
./sr check
./sr recite examples/krishna.txt -o out/krishna.wav

# Or activate venv, then use python
source .venv/bin/activate
python -m sanskrit_reciter check
python -m sanskrit_reciter recite examples/krishna.txt -o out/krishna.wav

# multi-verse → one continuous audio
./sr recite path/to/verses.txt -o out/verse.wav --all

# large corpus (skip failures)
./sr recite examples/amarakosha_brahma_varga.txt \
  -o out/brahma_varga.wav --all --continue-on-error
```

Bare `./sr FILE …` still works (defaults to `recite`).

## Learning pack (HTML + per-śloka audio)

Builds a **self-contained study folder**: each śloka, then its recitation player, with hooks for future anvaya / pratipadārtha / grammar / literary notes.

```bash
./sr learn examples/krishna.txt \
  -o out/learn_krishna \
  --title "Kṛṣṇa maṅgalācaraṇa"

# with sample annotation decorations
./sr learn examples/learn_demo.txt \
  -o out/learn_demo \
  --annotations examples/sample_annotations.json \
  --title "Demo learning pack"

# Amarakośa brahmavarga (continues on errors by default)
./sr learn examples/amarakosha_brahma_varga.txt \
  -o out/learn_brahma_varga \
  --title "अमरकोश · ब्रह्मवर्ग" \
  --continue-on-error
```

Open `out/learn_…/index.html` in a browser (works offline). Pack layout:

```
out/learn_demo/
  index.html          # study UI (Play all, auto-advance)
  manifest.json       # machine-readable pack (extensible)
  audio/001.wav …
  assets/             # reserved for future media
```

### Extending with literary annotations

Optional JSON sidecar (`--annotations`) attaches decorations per verse id (`001`, `002`, …):

| `kind` | Use |
|--------|-----|
| `anvaya` | Prose word-order |
| `pratipada` | Word-by-word meanings (table) |
| `grammar` | Vyākaraṇa notes |
| `literary` | Metre, alankāra, rasa |
| `translation` | Free translation |
| `notes` | Free teacher notes |

Custom `kind` strings are allowed. See `examples/sample_annotations.json`. Re-render HTML later without TTS by editing `manifest.json` and calling the builder’s `rebuild_html_only` helper.

## Publish to GitHub Pages (`docs/`)

Learning packs live under **`docs/`** in this repo and are served as a static site.

```bash
# Build a pack (local TTS)
./sr learn examples/amarakosha_brahma_varga.txt \
  -o out/learn_brahma_varga \
  --title "अमरकोश · ब्रह्मवर्ग"

# Copy into docs/<slug>/ and refresh the catalog
./sr publish out/learn_brahma_varga --slug brahma-varga
./sr publish out/learn_demo --slug demo

# Inspect
./sr publish --list
open docs/index.html   # local preview of the catalog
```

**Enable Pages once** (repo on GitHub):

1. **Settings → Pages → Source: GitHub Actions** (uses `.github/workflows/pages.yml`), **or**
2. Source: **Deploy from a branch** → `main` → folder **`/docs`**

Then:

```bash
git add docs .github/workflows/pages.yml
git commit -m "Publish learning packs to GitHub Pages"
git push
```

Public URLs (project site):

```text
https://<user>.github.io/<repo>/
https://<user>.github.io/<repo>/brahma-varga/
https://<user>.github.io/<repo>/demo/
```

Already published in-tree: `docs/demo/`, `docs/brahma-varga/` (~28 MB including audio). See `docs/README.md`.

### Input format

UTF-8 text, any major Indian script. Prefer complete verses with daṇḍas:

```
वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।
देवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥
```

Multiple verses: separate with `॥` or blank lines.

## Layout

```
sanskrit_reciter/     CLI + local engine
vendor/vagdhenu/      Vāgdhenu text frontend + meter + reference bank
vendor/BigVGAN/       NVIDIA BigVGAN code
models/               Downloaded weights (voice, vocoder, vocos) — gitignored
examples/             Sample Sanskrit verses
scripts/
  setup_local.sh      One-shot setup
  download_models.py  Fetch weights into models/
```

## How it works (local)

1. Read text file, split into shlokas  
2. Detect meter (chandas) from the verse  
3. Sanskrit text frontend (Devanagari → Kannada route, sandhi, visarga…)  
4. Vāgdhenu DiT (IndicF5/F5-TTS) + fine-tuned BigVGAN on device  
5. Write 24 kHz mono WAV  

All checkpoints live under `models/`. Runtime sets `HF_HUB_OFFLINE=1` so nothing is downloaded.

## CLI

### Commands

| Command | Meaning |
|---------|---------|
| `recite TEXT -o out.wav` | Chant audio |
| `learn TEXT -o pack_dir/` | HTML learning pack |
| `publish PACK --slug NAME` | Copy pack → `docs/` for GitHub Pages |
| `check` | Verify local weights |

### Shared / recite flags

| Flag | Meaning |
|------|---------|
| `--all` | `recite`: stitch all ślokas into one WAV |
| `--gap S` | Silence between verses (default 0.75) |
| `--save-parts` | Also write per-verse WAVs |
| `--continue-on-error` | Skip failed verses |
| `--strict` | Stop on first failure |
| `--meter` / `--seed` / `--device` / `--nfe` | Synthesis controls |
| `--list-shlokas` | Parse only |
| `--annotations FILE` | `learn`: JSON decorations |
| `--title` / `--subtitle` | `learn`: pack metadata |
| `--skip-audio` | `learn`: HTML only (no TTS) |
| `--slug` | `publish`: URL folder under `docs/` |
| `--list` / `--catalog-only` | `publish`: list packs / rebuild index |

## Attribution

- **Vāgdhenu**: [prathoshap/vagdhenu](https://github.com/prathoshap/vagdhenu) (Apache-2.0)  
- IndicF5, NVIDIA BigVGAN-v2, F5-TTS — see their licenses  
- Intended for study / pārāyaṇa / accessibility; do not impersonate  

## License

Apache-2.0 for this wrapper. Upstream model terms apply to the weights.
