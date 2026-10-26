"""Parity check: our LateOn encoder (altembed.MultiVectorEncoder) vs the official ONNX export. Run: uv run python scripts/check_lateon_onnx.py"""

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download

from docsearch.altembed import load_encoder
from docsearch.config import load_config

cfg = load_config()
enc = load_encoder(cfg, "lateon")
spec = cfg["models"]["embedders"]["lateon"]
d = snapshot_download(spec["id"], revision=spec.get("revision"))
s = ort.InferenceSession(f"{d}/model.onnx")
for text, q in [
    ("What drove IBM Z growth in 2020?", True),
    ("Systems revenue: IBM Z and Storage growth driven by product cycle, up 12%.", False),
]:
    mine = enc._encode([text], q)[0].astype(np.float32)
    t = enc.tok([text], max_length=31, truncation=True, return_tensors="np")
    ids = np.concatenate([t["input_ids"][:, :1], [[enc.q_id if q else enc.d_id]], t["input_ids"][:, 1:]], 1).astype(
        np.int64
    )
    out = s.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    if not q:
        out = out[~np.isin(ids[0], sorted(enc.skip))]
    print(
        "query" if q else "doc",
        mine.shape,
        out.shape,
        "max abs diff",
        float(np.abs(mine - out).max()),
        "min cos",
        float((mine * out).sum(1).min()),
    )
