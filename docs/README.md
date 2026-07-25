# GitHub Pages site

This folder is the public site for **SanskritReciter** learning packs.

## Layout

```
docs/
  index.html          # catalog of packs
  catalog.json        # machine-readable list
  .nojekyll
  <slug>/             # one learning pack each
    index.html
    manifest.json
    audio/*.wav
```

## Publish a pack

```bash
# 1. Build locally (needs models / venv)
./sr learn examples/amarakosha_brahma_varga.txt \
  -o out/learn_brahma_varga \
  --title "अमरकोश · ब्रह्मवर्ग"

# 2. Copy into docs/
./sr publish out/learn_brahma_varga --slug brahma-varga

# 3. Commit and push
git add docs
git commit -m "Publish brahma-varga learning pack"
git push
```

## Enable Pages (once)

**Option A — GitHub Actions (recommended)**  
Repo → **Settings → Pages → Build and deployment → Source: GitHub Actions**  
Pushing changes under `docs/` runs `.github/workflows/pages.yml`.

**Option B — Deploy from branch**  
Source: **Deploy from a branch** → branch `main` → folder `/docs`.

Site URL (project pages):

```text
https://<user>.github.io/<repo>/
https://<user>.github.io/<repo>/brahma-varga/
```

Audio is large; only publish packs you intend to share.
