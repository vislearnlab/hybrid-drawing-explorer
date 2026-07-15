#!/usr/bin/env python3
"""
Build drawings/ + points.json for the Hybrid Drawing Explorer.

Data, rebuilt from MongoDB (the committed analysis CSVs are anonymized):

  * HYBRID drawings  — kiddraw.cdm_hybrid_v1, the 8 "usable" hybrid concepts
    whose constituents both exist as pure kiddraw concepts
    (train cat, bike bee, mushroom house, rabbit boat, sheep fish,
     dinosaur truck, elephant snail, tiger frog). Each finalImage carries `age`.
  * PURE constituent drawings — the 16 constituents, pulled from the kiddraw
    cdm_run_v3..v8 collections (the ~37k object-drawing dataset).

Quality filters mirror exploratory-drawing-analysis/01_data_preparation:
  - drop control items (square / shape / a dog / …)
  - draw_duration = (endTrialTime - startTrialTime)/1000  >= 2.0 s
  - ink_density = fraction of pixels < 250 on white  <= 0.15  (drop scribbles)
  - dedup to one finalImage per (sessionId, category), keeping the latest
Then sample per concept to keep the page light. Transparent PNGs are saved.

For each hybrid drawing d with constituent centroids c1, c2 (mean pure embedding):
  w        = clip01((d-c1)·(c2-c1)/|c2-c1|²)   residual = ||d-(c1+w(c2-c1))||
Both UMAP(cosine) and t-SNE layouts are computed (toggle in the page).

    SEA_MONGO_URI='mongodb://user:pass@host:27017/?authSource=admin' python3 build_data.py
"""
import os, io, json, base64, re, random

import numpy as np
from PIL import Image
import umap
from sklearn.manifold import TSNE

import clip_lib

random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
DRAW_DIR = os.path.join(HERE, "drawings")
DB = "kiddraw"
HYBRID_COLLS = ["cdm_hybrid_v1", "SONA_hybrid_run_v1"]   # both hybrid collections
PURE_COLLS = ["cdm_run_v8", "cdm_run_v7", "cdm_run_v6", "cdm_run_v5", "cdm_run_v4",
              "cdm_run_v3", "cdm_run_v2", "Bing_run_v4", "india_run_v1", "SONA_hybrid_run_v1"]
CONTROLS = {"square", "shape", "this square", "copied_square", "traced_square"}

# all 17 hybrids across both experiment stim sets, with explicit constituents
# (explicit map needed because "an ice cream hat" doesn't word-split cleanly).
HYBRID_MAP = {
    "a train cat": ("a train", "a cat"), "a bike bee": ("a bike", "a bee"),
    "a mushroom house": ("a mushroom", "a house"), "a rabbit boat": ("a rabbit", "a boat"),
    "a sheep fish": ("a sheep", "a fish"), "a dinosaur truck": ("a dinosaur", "a truck"),
    "an elephant snail": ("an elephant", "a snail"), "a tiger frog": ("a tiger", "a frog"),
    "a cow whale": ("a cow", "a whale"), "a bear tree": ("a bear", "a tree"),
    "an airplane lamp": ("an airplane", "a lamp"), "a horse car": ("a horse", "a car"),
    "an ice cream hat": ("an ice cream", "a hat"), "a phone bird": ("a phone", "a bird"),
    "a spider watch": ("a spider", "a watch"), "an octopus cup": ("an octopus", "a cup"),
    "a camel dog": ("a camel", "a dog"),
}
USABLE_HYBRIDS = list(HYBRID_MAP)
DUR_MIN, INK_MIN, INK_MAX = 2.0, 0.01, 0.15   # drop blank (<1% ink) and scribbles
SAMPLE_HYBRID, SAMPLE_PURE = 200, 110     # displayed per concept
CENTROID_SAMPLE = 500                      # drawings per concept for the FYP centroid (embed-only)
EMB_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".emb_cache.npz")


def constituents(h):
    return HYBRID_MAP[h]


def age_num(a):
    if a is None:
        return None
    if str(a).lower() == "adult":
        return 18
    m = re.search(r"(\d+)", str(a))
    return int(m.group(1)) if m else None


def mongo():
    from pymongo import MongoClient
    uri = os.environ.get("SEA_MONGO_URI")
    if not uri and os.path.exists(os.path.join(HERE, "auth.txt")):
        uri = open(os.path.join(HERE, "auth.txt")).readline().strip()
    if not uri:
        raise SystemExit("Set SEA_MONGO_URI or auth.txt")
    return MongoClient(uri, serverSelectionTimeoutMS=10000)[DB]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def decode(doc):
    raw = doc["imgData"]
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")


def ink_density(rgba):
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    g = np.asarray(bg.convert("L"))
    return float((g < 250).sum()) / g.size


def collect(colls, categories, sample_n, kind):
    """Aggregate across one or more collections. Dedup latest per
    (sessionId,category) with the duration filter (cheap, no decode); then per
    category shuffle and lazily decode candidates, keeping ones whose ink
    density is in [INK_MIN, INK_MAX] (drops blanks and scribbles), up to
    sample_n. Never decodes the whole collection."""
    if not isinstance(colls, (list, tuple)):
        colls = [colls]
    by_cat = {c: {} for c in categories}
    for coll in colls:
        cur = coll.find({"dataType": "finalImage", "category": {"$in": list(categories)}},
                        {"imgData": 1, "category": 1, "sessionId": 1, "trialNum": 1,
                         "startTrialTime": 1, "endTrialTime": 1, "age": 1})
        for d in cur:
            st, et = d.get("startTrialTime"), d.get("endTrialTime")
            if st and et and (et - st) / 1000.0 < DUR_MIN:   # only filter when timed
                continue
            prev = by_cat[d["category"]].get(d.get("sessionId"))
            if prev is None or (d.get("endTrialTime") or 0) >= (prev.get("endTrialTime") or 0):
                by_cat[d["category"]][d.get("sessionId")] = d
    out = []
    for cat, docs in by_cat.items():
        cands = list(docs.values()); random.shuffle(cands)
        kept = 0
        for d in cands:
            if kept >= sample_n:
                break
            try:
                rgba = decode(d)
                dink = ink_density(rgba)
                if dink < INK_MIN or dink > INK_MAX:     # blank or scribble
                    continue
            except Exception:
                continue
            fname = f"{slug(cat)}_{slug(d.get('sessionId'))}_{d.get('trialNum')}.png"
            rgba.save(os.path.join(DRAW_DIR, fname))     # transparent PNG
            out.append(dict(cat=cat, file=fname, kind=kind, age=age_num(d.get("age"))))
            kept += 1
    return out


def extra_embeddings(colls, concept, need, skip_files, cache):
    """Embed-only pool for a concept's centroid: pull up to `need` more quality
    drawings (deduped, not already displayed), embed them without saving PNGs.
    Cached by filename so re-runs are fast."""
    if need <= 0:
        return []
    by_sess = {}
    for coll in colls:
        for d in coll.find({"dataType": "finalImage", "category": concept},
                           {"imgData": 1, "sessionId": 1, "trialNum": 1,
                            "startTrialTime": 1, "endTrialTime": 1}):
            st, et = d.get("startTrialTime"), d.get("endTrialTime")
            if st and et and (et - st) / 1000.0 < DUR_MIN:
                continue
            s = d.get("sessionId")
            prev = by_sess.get(s)
            if prev is None or (d.get("endTrialTime") or 0) >= (prev.get("endTrialTime") or 0):
                by_sess[s] = d
    cands = list(by_sess.values()); random.shuffle(cands)
    out = []
    for d in cands:
        if len(out) >= need:
            break
        fname = f"{slug(concept)}_{slug(d.get('sessionId'))}_{d.get('trialNum')}.png"
        if fname in skip_files:
            continue
        if fname in cache:
            out.append(cache[fname]); continue
        try:
            rgba = decode(d); dink = ink_density(rgba)
            if dink < INK_MIN or dink > INK_MAX:
                continue
            e = clip_lib.embed_pil(rgba)
        except Exception:
            continue
        cache[fname] = e; out.append(e)
    return out


def scale(v, lo, hi, a=40, b=960):
    return round(a + (v - lo) / (hi - lo + 1e-9) * (b - a), 2)


def main():
    os.makedirs(DRAW_DIR, exist_ok=True)
    db = mongo()
    pures = sorted({c for h in USABLE_HYBRIDS for c in constituents(h)})
    print(f"{len(USABLE_HYBRIDS)} hybrids, {len(pures)} pure constituents")

    # ---- hybrids (from both hybrid collections) ----
    items = collect([db[c] for c in HYBRID_COLLS], set(USABLE_HYBRIDS), SAMPLE_HYBRID, 1)
    print(f"hybrids after filter+sample: {len(items)}")

    # ---- pures: aggregate across kiddraw runs until each concept hits the cap ----
    need = {p: SAMPLE_PURE for p in pures}
    pure_items = []
    for cn in PURE_COLLS:
        want = [p for p, k in need.items() if k > 0]
        if not want:
            break
        got = collect(db[cn], set(want), max(need.values()), 0)
        for it in got:
            if need[it["cat"]] > 0:
                pure_items.append(it); need[it["cat"]] -= 1
        print(f"  {cn}: +{len(got)} (remaining need: {sum(need.values())})")
    items += pure_items
    print(f"total drawings: {len(items)}  ({sum(i['kind'] for i in items)} hybrid)")

    # ---- embed (cached by filename so re-runs after a filter tweak are fast) ----
    cache = {}
    if os.path.exists(EMB_CACHE):
        z = np.load(EMB_CACHE, allow_pickle=True)
        cache = {f: e for f, e in zip(z["files"], z["embs"])}
    embs, hits = [], 0
    for j, it in enumerate(items):
        f = it["file"]
        if f in cache:
            embs.append(cache[f]); hits += 1
        else:
            e = clip_lib.image_embedding(os.path.join(DRAW_DIR, f)); cache[f] = e; embs.append(e)
        if (j + 1) % 300 == 0:
            print(f"    embedded {j+1}/{len(items)} ({hits} cached)")
    E = np.array(embs)
    print(f"  {hits}/{len(items)} display embeddings from cache")

    # ---- FYP-style centroids: mean over MANY kiddraw drawings per concept ----
    # (embed-only pool, not saved to disk, so the repo stays light)
    cent = {}
    display_files = {it["file"] for it in items}
    pure_colls = [db[cn] for cn in PURE_COLLS]
    for p in pures:
        pool = [E[i] for i, it in enumerate(items) if it["kind"] == 0 and it["cat"] == p]
        pool += extra_embeddings(pure_colls, p, CENTROID_SAMPLE - len(pool), display_files, cache)
        v = np.mean(pool, axis=0); cent[p] = v / (np.linalg.norm(v) + 1e-9)
        print(f"    centroid {p}: {len(pool)} drawings")
    np.savez(EMB_CACHE, files=np.array(list(cache.keys())), embs=np.array(list(cache.values())))
    # matrix of all constituent centroids for k-way rank-ordering (the FYP metric)
    concept_names = list(cent.keys())
    cmat = np.array([cent[c] for c in concept_names])          # (K, 512), unit rows
    for i, it in enumerate(items):
        if it["kind"] == 0:
            it.update(c1="", c2="", w=None, residual=None, sim1=None, sim2=None, hyb=-1,
                      rank1=None, rank2=None, both_top5=None, n_top5=None,
                      both_top10=None, n_top10=None)
            continue
        a1, a2 = constituents(it["cat"]); c1, c2 = cent[a1], cent[a2]; d = E[i]
        v = c2 - c1; t = float((d - c1) @ v / (v @ v + 1e-9)); tc = min(1, max(0, t))
        # k-way: rank every constituent concept by similarity to this drawing
        sims = d @ cmat.T
        order = np.argsort(-sims)
        rank_of = {concept_names[order[r]]: r + 1 for r in range(len(concept_names))}
        r1, r2 = rank_of[a1], rank_of[a2]
        it.update(c1=a1, c2=a2, hyb=USABLE_HYBRIDS.index(it["cat"]),
                  w=round(tc, 4), residual=round(float(np.linalg.norm(d - (c1 + tc * v))), 4),
                  sim1=round(float(d @ c1), 4), sim2=round(float(d @ c2), 4),
                  rank1=r1, rank2=r2,
                  both_top5=int(r1 <= 5 and r2 <= 5), n_top5=int(r1 <= 5) + int(r2 <= 5),
                  both_top10=int(r1 <= 10 and r2 <= 10), n_top10=int(r1 <= 10) + int(r2 <= 10))

    # ---- layouts: UMAP + t-SNE ----
    print("UMAP ...")
    u = umap.UMAP(n_neighbors=15, min_dist=0.12, metric="cosine", random_state=42).fit_transform(E)
    print("t-SNE ...")
    Xz = (E - E.mean(0)) / (E.std(0) + 1e-9)
    ts = TSNE(n_components=2, perplexity=40, init="pca", learning_rate=200.0, random_state=0).fit_transform(Xz)

    resid = [it["residual"] for it in items if it["kind"] == 1]
    rlo, rhi = min(resid), max(resid)
    P = dict(file=[], cat=[], kind=[], hyb=[], age=[], c1=[], c2=[], w=[], residual=[],
             sim1=[], sim2=[], rank1=[], rank2=[], both_top5=[], n_top5=[],
             both_top10=[], n_top10=[], ux=[], uy=[], tx=[], ty=[], ix=[], iy=[])
    for i, it in enumerate(items):
        for k in ("file", "cat", "kind", "hyb", "age", "c1", "c2", "w", "residual",
                  "sim1", "sim2", "rank1", "rank2", "both_top5", "n_top5",
                  "both_top10", "n_top10"):
            P[k].append(it[k])
        P["ux"].append(scale(u[i, 0], u[:, 0].min(), u[:, 0].max()))
        P["uy"].append(scale(u[i, 1], u[:, 1].min(), u[:, 1].max()))
        P["tx"].append(scale(ts[i, 0], ts[:, 0].min(), ts[:, 0].max()))
        P["ty"].append(scale(ts[i, 1], ts[:, 1].min(), ts[:, 1].max()))
        if it["kind"] == 1:
            P["ix"].append(scale(it["w"], 0, 1)); P["iy"].append(scale(it["residual"], rlo, rhi, 960, 40))
        else:
            P["ix"].append(None); P["iy"].append(None)
    P["n"] = len(items)

    out = dict(draw_dir="drawings", hybrids=USABLE_HYBRIDS, pures=pures,
               constituents={h: list(constituents(h)) for h in USABLE_HYBRIDS},
               n_concepts=len(pures),
               n_hybrid=sum(P["kind"]), n_pure=P["n"] - sum(P["kind"]), items=P)
    json.dump(out, open(os.path.join(HERE, "points.json"), "w"))
    print(f"wrote points.json: {P['n']} ({out['n_hybrid']} hybrid + {out['n_pure']} pure), "
          f"{len(os.listdir(DRAW_DIR))} PNGs")


if __name__ == "__main__":
    main()
