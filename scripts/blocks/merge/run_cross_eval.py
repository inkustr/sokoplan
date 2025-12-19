from __future__ import annotations

"""
Run cross-evaluation for candidate (model_pack, data_pack) pairs.

Reads pairs CSV (model_pack,data_pack) and produces results CSV with:
  model_pack,data_pack,n,mae,mse,rmse,r2

Run:
  source .venv/bin/activate
  python -m scripts.blocks.merge.run_cross_eval \
    --pairs results/merge_pairs.csv \
    --labels_dir data/packs_labels_festival \
    --models_dir artifacts/packs_models \
    --out results/cross_eval.csv \
    --limit_records 5000 \
    --num_shards 20 --shard_idx 0
"""

import argparse
import csv
import os
import tempfile


def _iter_pairs(pairs_path: str):
    with open(pairs_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row["model_pack"], row["data_pack"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--labels_dir", default="data/packs_labels_festival")
    p.add_argument("--models_dir", default="artifacts/packs_models")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--limit_records", type=int, default=5000)
    p.add_argument("--num_shards", type=int, default=28)
    p.add_argument("--shard_idx", type=int, default=0)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    from scripts.blocks.split.eval_pack_model import main as _eval_main 
    import sys
    import json

    pairs = list(_iter_pairs(args.pairs))
    pairs = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard_idx]

    tmpdir = tempfile.mkdtemp(prefix="cross_eval_")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_pack", "data_pack", "n", "mae", "mse", "rmse", "r2"])

        for model_pack, data_pack in pairs:
            ckpt = os.path.join(args.models_dir, f"{model_pack}_best.pt")
            labels = os.path.join(args.labels_dir, f"{data_pack}.jsonl")
            if not os.path.exists(ckpt) or not os.path.exists(labels):
                continue

            tmp_out = os.path.join(tmpdir, f"{model_pack}__{data_pack}.json")
            argv = [
                "eval_pack_model",
                "--ckpt",
                ckpt,
                "--labels",
                labels,
                "--out",
                tmp_out,
                "--limit_records",
                str(args.limit_records),
            ]
            if args.device:
                argv += ["--device", args.device]

            old_argv = sys.argv
            try:
                sys.argv = argv
                _eval_main()
            finally:
                sys.argv = old_argv

            with open(tmp_out, "r", encoding="utf-8") as jf:
                obj = json.load(jf)
            m = obj["overall"]
            w.writerow(
                [
                    model_pack,
                    data_pack,
                    int(m.get("n", 0)),
                    float(m.get("mae", 0.0)),
                    float(m.get("mse", 0.0)),
                    float(m.get("rmse", 0.0)),
                    float(m.get("r2", 0.0)),
                ]
            )


if __name__ == "__main__":
    main()


