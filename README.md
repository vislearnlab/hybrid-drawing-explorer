# Hybrid Drawing Explorer

An interactive web page for exploring children's drawings of **hybrid concepts** —
blends of two familiar things — 17 of them (bike bee, tiger frog, ice cream hat, octopus cup, …) — alongside the pure constituent drawings, in CLIP
embedding space. Companion to
[`exploratory-drawing-analysis`](https://github.com/vislearnlab/exploratory-drawing-analysis),
built in the style of the lab's other drawing explorers.

Two linked panels, side by side:

- **🗺️ UMAP** — every drawing (hybrids + pure constituents) laid out by UMAP of
  its CLIP (ViT-B-32) embedding, so you can see where hybrids sit relative to
  the pure concepts they blend.
- **⟷ Interpolation** — each hybrid drawing placed on a **constituent-1 ↔
  constituent-2** axis (how much it leans toward each) against its **residual
  distance** (how far it sits *off the line* between the two centroids — a proxy
  for emergent features that aren't a simple blend).

Hover a point in either panel and the **same drawing** lights up in the other,
with a card showing the drawing, its blend weight, residual, and similarity to
each constituent.

## What the metrics mean

For a hybrid drawing with CLIP embedding **d** and its two constituent centroids
**c₁**, **c₂** (mean embedding of all pure drawings of each concept):

- **interpolation weight** `w = clip01( (d−c₁)·(c₂−c₁) / |c₂−c₁|² )` — 0 if the
  drawing projects onto **c₁**, 1 onto **c₂**, 0.5 at the midpoint.
- **residual** `‖ d − (c₁ + w·(c₂−c₁)) ‖` — distance from the line between the
  two constituents; high = features beyond a linear blend.
- **sim1 / sim2** — cosine similarity to each constituent centroid.

These mirror the interpolation / residual analyses in the source repo.

**Color by:** hybrid vs pure, hybrid category, interpolation weight, residual, or
which constituent the drawing favors. **Filter** the hybrid categories and toggle
the pure constituents.

## Run locally

```bash
python3 -m http.server 8000
# open http://127.0.0.1:8000/
```

## Files

- `index.html` — the self-contained explorer (no build step).
- `points.json` — UMAP + interpolation coords and per-drawing scores.
- `drawings/` — the drawing PNGs (150×150, flattened onto white).
- `clip_lib.py` — loads OpenCLIP ViT-B-32 and embeds images.
- `build_data.py` — regenerates everything from MongoDB (see below).

## Rebuilding the data

Needs `numpy`, `pillow`, `pymongo`, `umap-learn`, and `open_clip_torch`. The
drawings live in the lab's `kiddraw.SONA_hybrid_run_v1` MongoDB collection (the
committed analysis CSVs in the source repo are anonymized and don't map back to
images, so this rebuilds metrics directly from the embeddings).

```bash
export SEA_MONGO_URI='mongodb://USER:PASS@HOST:27017/?authSource=admin'  # do not commit
python3 build_data.py
```

Credentials can instead go in a git-ignored `auth.txt` (first line = connection
string).

**Note:** per-drawing *age* is not included — it exists only in the anonymized
analysis CSV, which can't be linked back to the Mongo images. If a non-anonymized
id mapping becomes available, age can be added.

## Data & paper

Hybrid drawing task and analyses:
[`exploratory-drawing-analysis`](https://github.com/vislearnlab/exploratory-drawing-analysis)
(vislearnlab).
