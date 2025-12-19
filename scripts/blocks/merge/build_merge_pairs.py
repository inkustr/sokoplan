from __future__ import annotations

"""
Build candidate cross-evaluation pairs for merging.
For each pack, pick K nearest packs in a simple pack-level stats space and emit directed pairs (model_pack, data_pack) for evaluation.

Run:
  source .venv/bin/activate
  python -m scripts.blocks.merge.build_merge_pairs --labels_dir data/packs_labels_festival --out results/merge_pairs.csv --k 10 --sample_records 2000
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _iter_jsonl(path: str, limit: int):
    with open(path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i >= limit:
                return
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def _pack_vector(labels_path: str, sample_records: int) -> np.ndarray:
    ws = []
    hs = []
    bs = []
    gs = []
    ys = []
    for rec in _iter_jsonl(labels_path, sample_records):
        ws.append(int(rec["width"]))
        hs.append(int(rec["height"]))
        bs.append(int(int(rec["boxes"]).bit_count()))
        gs.append(int(int(rec["goals"]).bit_count()))
        ys.append(float(int(rec["y"])))
    if not ys:
        return np.zeros(6, dtype=float)
    yarr = np.array(ys, dtype=float)
    return np.array(
        [
            float(np.mean(ws)),
            float(np.mean(hs)),
            float(np.mean(bs)),
            float(np.mean(gs)),
            float(np.mean(yarr)),
            float(np.var(yarr)),
        ],
        dtype=float,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir", default="data/packs_labels_festival")
    p.add_argument("--out", default="results/merge_pairs.csv")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--sample_records", type=int, default=2000)
    args = p.parse_args()

    files = sorted([f for f in os.listdir(args.labels_dir) if f.endswith(".jsonl")])
    if not files:
        raise SystemExit(f"No jsonl in {args.labels_dir}")

    packs = [os.path.splitext(f)[0] for f in files]
    vecs = []
    for f in files:
        vecs.append(_pack_vector(os.path.join(args.labels_dir, f), args.sample_records))

    X = np.stack(vecs, axis=0)
    # standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    Xn = (X - mean) / std

    nn = NearestNeighbors(n_neighbors=min(args.k + 1, len(packs)), metric="euclidean")
    nn.fit(Xn)
    dists, idxs = nn.kneighbors(Xn, return_distance=True)

    pairs: List[Tuple[str, str]] = []
    for i, neigh in enumerate(idxs):
        src = packs[i]
        for j in neigh[1:]:  # skip itself
            pairs.append((src, packs[int(j)]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_pack", "data_pack"])
        for a, b in pairs:
            w.writerow([a, b])

    print(f"packs={len(packs)} k={args.k} pairs={len(pairs)} → {args.out}")


if __name__ == "__main__":
    main()
