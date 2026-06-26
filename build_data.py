#!/usr/bin/env python3
"""
Build drawings/ + points.json for the Hybrid Drawing Explorer.

Children were asked to draw HYBRID concepts ("a bike bee", "a tiger frog", ...)
— each a blend of two familiar concepts — plus the pure constituent concepts.
This explorer shows, side by side:

  * a UMAP of CLIP embeddings of every drawing (hybrids + pure constituents), and
  * an interpolation view: each hybrid drawing placed on a constituent-1 ↔
    constituent-2 axis (how much it blends the two) vs its residual distance
    (how far it sits off the line between them — i.e. emergent features).

Everything is rebuilt from the `kiddraw.SONA_hybrid_run_v1` MongoDB collection
(the committed analysis CSVs are anonymized and don't map back to images):

  1. pull every finalImage for the 16 hybrid categories + their pure constituents
  2. CLIP-embed each (OpenCLIP ViT-B-32) and save the PNG
  3. pure-concept centroid = mean adult/child embedding per concept
  4. per hybrid drawing d with constituent centroids c1, c2:
       w        = clip01( (d-c1)·(c2-c1) / |c2-c1|² )     (interpolation weight)
       residual = || d - (c1 + w·(c2-c1)) ||              (off-line distance)
       sim1,sim2= cosine(d, c1), cosine(d, c2)
  5. UMAP(cosine) of all embeddings → 2-D map

Mongo creds via SEA_MONGO_URI or auth.txt (git-ignored).

    SEA_MONGO_URI='mongodb://user:pass@host:27017/?authSource=admin' python3 build_data.py
"""
import os, io, csv, json, base64, re

import numpy as np
from PIL import Image
import umap

import clip_lib

HERE = os.path.dirname(os.path.abspath(__file__))
DRAW_DIR = os.path.join(HERE, "drawings")
DB_NAME, COLL = "kiddraw", "SONA_hybrid_run_v1"
EXCLUDE = {"an ice cream"}          # 3 tokens but a single concept, not a hybrid


def article(word):
    return ("an " if word[0].lower() in "aeiou" else "a ") + word


def constituents(hybrid_cat):
    """'a bike bee' -> ('a bike', 'a bee'); 'an elephant snail' -> ('an elephant','a snail')."""
    toks = hybrid_cat.split()
    return article(toks[1]), article(toks[2])


def mongo_coll():
    from pymongo import MongoClient
    uri = os.environ.get("SEA_MONGO_URI")
    if not uri and os.path.exists(os.path.join(HERE, "auth.txt")):
        uri = open(os.path.join(HERE, "auth.txt")).readline().strip()
    if not uri:
        raise SystemExit("Set SEA_MONGO_URI or create auth.txt with the connection string.")
    return MongoClient(uri, serverSelectionTimeoutMS=10000)[DB_NAME][COLL]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def save_png(doc, path):
    raw = doc["imgData"]
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    bg.convert("RGB").save(path, "PNG")


def scale01(v, lo, hi, a=40, b=960):
    return a + (v - lo) / (hi - lo + 1e-9) * (b - a)


def main():
    os.makedirs(DRAW_DIR, exist_ok=True)
    col = mongo_coll()

    cats = col.distinct("category", {"dataType": "finalImage"})
    hybrids = sorted(c for c in cats if len(c.split()) == 3 and c not in EXCLUDE)
    pures = sorted(c for c in cats if len(c.split()) == 2)
    pureset = set(pures)
    # keep only hybrids whose two constituents are both present as pure concepts
    hybrids = [h for h in hybrids if all(c in pureset for c in constituents(h))]
    needed_pures = sorted({c for h in hybrids for c in constituents(h)})
    print(f"{len(hybrids)} hybrids, {len(needed_pures)} constituent pure concepts")

    # ---- pull + embed every drawing ----------------------------------------
    items = []   # dicts with kind/cat/file/emb
    seen = set()
    want = hybrids + needed_pures
    docs = col.find({"dataType": "finalImage", "category": {"$in": want}},
                    {"imgData": 1, "category": 1, "sessionId": 1, "trialNum": 1})
    for d in docs:
        cat = d["category"]
        key = (d.get("sessionId"), d.get("trialNum"), cat)
        if key in seen:
            continue
        seen.add(key)
        fname = f"{slug(cat)}_{slug(str(d.get('sessionId')))}_{d.get('trialNum')}.png"
        path = os.path.join(DRAW_DIR, fname)
        try:
            save_png(d, path)
            emb = clip_lib.image_embedding(path)
        except Exception as e:
            continue
        items.append(dict(cat=cat, file=fname, kind=1 if cat in set(hybrids) else 0, emb=emb))
        if len(items) % 200 == 0:
            print(f"    embedded {len(items)} drawings")
    print(f"{len(items)} drawings embedded")

    E = np.array([it["emb"] for it in items])

    # ---- pure-concept centroids (normalized mean) --------------------------
    cent = {}
    for c in needed_pures:
        idx = [i for i, it in enumerate(items) if it["kind"] == 0 and it["cat"] == c]
        v = E[idx].mean(0)
        cent[c] = v / (np.linalg.norm(v) + 1e-9)

    # ---- interpolation weight + residual for each hybrid -------------------
    for it in items:
        if it["kind"] == 0:
            it.update(c1="", c2="", w=None, residual=None, sim1=None, sim2=None)
            continue
        a1, a2 = constituents(it["cat"])
        c1, c2 = cent[a1], cent[a2]
        d = it["emb"]
        v = c2 - c1
        t = float((d - c1) @ v / (v @ v + 1e-9))
        tc = min(1.0, max(0.0, t))
        proj = c1 + tc * v
        it.update(c1=a1, c2=a2, w=round(tc, 4),
                  residual=round(float(np.linalg.norm(d - proj)), 4),
                  sim1=round(float(d @ c1), 4), sim2=round(float(d @ c2), 4))

    # ---- UMAP of all embeddings --------------------------------------------
    print("UMAP on", E.shape, "...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.12, metric="cosine", random_state=42)
    xy = reducer.fit_transform(E)
    ux = [round(scale01(v, xy[:, 0].min(), xy[:, 0].max()), 2) for v in xy[:, 0]]
    uy = [round(scale01(v, xy[:, 1].min(), xy[:, 1].max()), 2) for v in xy[:, 1]]

    # ---- interpolation-view coords (hybrids only) --------------------------
    resid = [it["residual"] for it in items if it["kind"] == 1]
    rlo, rhi = min(resid), max(resid)

    P = dict(file=[], cat=[], kind=[], hyb=[], c1=[], c2=[], w=[], residual=[],
             sim1=[], sim2=[], ux=[], uy=[], ix=[], iy=[])
    for i, it in enumerate(items):
        P["file"].append(it["file"]); P["cat"].append(it["cat"]); P["kind"].append(it["kind"])
        P["hyb"].append(hybrids.index(it["cat"]) if it["kind"] == 1 else -1)
        P["c1"].append(it["c1"]); P["c2"].append(it["c2"])
        P["w"].append(it["w"]); P["residual"].append(it["residual"])
        P["sim1"].append(it["sim1"]); P["sim2"].append(it["sim2"])
        P["ux"].append(ux[i]); P["uy"].append(uy[i])
        if it["kind"] == 1:
            P["ix"].append(round(scale01(it["w"], 0, 1), 2))
            P["iy"].append(round(scale01(it["residual"], rlo, rhi, 960, 40), 2))  # high residual = top
        else:
            P["ix"].append(None); P["iy"].append(None)
    P["n"] = len(items)

    out = dict(draw_dir="drawings", hybrids=hybrids, pures=needed_pures,
               n_hybrid=sum(P["kind"]), n_pure=P["n"] - sum(P["kind"]), items=P)
    with open(os.path.join(HERE, "points.json"), "w") as f:
        json.dump(out, f)
    print(f"wrote points.json: {P['n']} drawings ({out['n_hybrid']} hybrid + {out['n_pure']} pure), "
          f"{len(os.listdir(DRAW_DIR))} PNGs")


if __name__ == "__main__":
    main()
