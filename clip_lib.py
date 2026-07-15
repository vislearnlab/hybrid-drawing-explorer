"""Shared CLIP (OpenCLIP ViT-B-32, pretrained=openai) helpers — the exact model
the sea-animals-draw preprocessing pipeline uses. Loaded once, reused by
build_data.py for both image and text zero-shot classification."""
import os
# avoid the conda MKL / llvm-OpenMP double-runtime crash; must precede torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import math
import numpy as np
import torch
import open_clip
from PIL import Image

torch.set_num_threads(1)

CATEGORIES = ["crab", "octopus", "seahorse", "shark", "turtle", "whale"]
# text anchors for zero-shot classification
DRAW_ANCHORS = [f"a drawing of {a}" for a in
                ["a crab", "an octopus", "a seahorse", "a shark", "a turtle", "a whale"]]
TEXT_ANCHORS = ["a crab", "an octopus", "a seahorse", "a shark", "a turtle", "a whale"]

_M = {}


def _load():
    if _M:
        return _M
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # the "-quickgelu" variant matches the OpenAI pretrained weights exactly
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai")
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    _M.update(model=model, preprocess=preprocess, tok=tok, device=device,
              logit_scale=float(model.logit_scale.exp().item()))
    return _M


def _encode_text(texts):
    m = _load()
    with torch.no_grad():
        t = m["tok"](texts).to(m["device"])
        f = m["model"].encode_text(t)
        f = f / f.norm(dim=-1, keepdim=True)
    return f


def anchor_feats(which):
    return _encode_text(DRAW_ANCHORS if which == "draw" else TEXT_ANCHORS)


def embed_pil(img, content_crop=True):
    """L2-normalized CLIP embedding for an in-memory PIL image (any mode).
    Transparent images are flattened onto white first."""
    m = _load()
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        img = bg
    else:
        img = img.convert("RGB")
    if content_crop:
        gray = np.asarray(img.convert("L"))
        ys, xs = np.where(gray < 250)  # non-white ink
        if len(xs) and len(ys):
            pad = 6
            l, r = max(xs.min() - pad, 0), min(xs.max() + pad, img.width)
            t, b = max(ys.min() - pad, 0), min(ys.max() + pad, img.height)
            img = img.crop((l, t, r, b))
    with torch.no_grad():
        x = m["preprocess"](img).unsqueeze(0).to(m["device"])
        f = m["model"].encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()[0]


def image_embedding(path, content_crop=True):
    """L2-normalized CLIP embedding for a drawing PNG on disk."""
    return embed_pil(Image.open(path), content_crop=content_crop)


def _chunk(text, max_len=50):
    """Split long text into word chunks (CLIP's 77-token limit ~ 50 words)."""
    words = text.split()
    if len(words) <= max_len:
        return [text]
    n = math.ceil(len(words) / max_len)
    size = math.ceil(len(words) / n)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def text_embedding(text):
    """L2-normalized CLIP text embedding (512-d numpy), averaged over chunks."""
    feats = _encode_text(_chunk(text))        # (k, d), already L2-normalized
    f = feats.mean(dim=0, keepdim=True)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()[0]


def prototype_probs(emb, protos):
    """6-way probabilities for an item: cosine to each adult prototype, then a
    self-calibrating (z-scored) softmax so probs are graded, not saturated."""
    sims = emb @ protos.T
    z = (sims - sims.mean()) / (sims.std() + 1e-9)
    p = np.exp(z)
    return p / p.sum()
