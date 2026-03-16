from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv_success_rate(path: Path) -> Tuple[float, int]:
    total = 0
    solved = 0
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("level_id"):
                continue
            total += 1
            solved += 1 if row.get("success") == "True" else 0
    if total == 0:
        return float("nan"), 0
    return solved / total, total


def _collect_matrix(raw_dir: Path) -> Tuple[np.ndarray, List[str], List[str], np.ndarray]:
    target_dirs = sorted([p for p in raw_dir.glob("target_group_*") if p.is_dir()])
    if not target_dirs:
        raise SystemExit(f"No target_group_* directories found in: {raw_dir}")

    targets = [p.name.replace("target_", "") for p in target_dirs]
    models_set = set()
    for td in target_dirs:
        for f in td.glob("model_group_*.csv"):
            models_set.add(f.stem.replace("model_", ""))
    if not models_set:
        raise SystemExit(f"No model_group_*.csv files found under: {raw_dir}")
    models = sorted(models_set)

    model_to_idx = {m: i for i, m in enumerate(models)}
    mat = np.full((len(targets), len(models)), np.nan, dtype=float)
    n_levels = np.zeros(len(targets), dtype=int)

    for i, td in enumerate(target_dirs):
        for f in sorted(td.glob("model_group_*.csv")):
            model = f.stem.replace("model_", "")
            j = model_to_idx[model]
            sr, n = _read_csv_success_rate(f)
            mat[i, j] = sr
            n_levels[i] = max(n_levels[i], n)
    return mat, targets, models, n_levels


def _diagonal_ranks(mat: np.ndarray, targets: List[str], models: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    model_to_idx = {m: i for i, m in enumerate(models)}
    ranks = np.full(len(targets), np.nan, dtype=float)
    gaps = np.full(len(targets), np.nan, dtype=float)

    for i, tg in enumerate(targets):
        if tg not in model_to_idx:
            continue
        row = mat[i, :]
        if np.all(np.isnan(row)):
            continue
        diag_j = model_to_idx[tg]
        diag_val = row[diag_j]
        if np.isnan(diag_val):
            continue
        valid = row[~np.isnan(row)]
        # rank 1 is best. "min" tie-handling: if equal to best, rank is 1.
        rank = 1 + np.sum(valid > diag_val)
        ranks[i] = float(rank)
        gaps[i] = float(np.nanmax(row) - diag_val)
    return ranks, gaps


def _row_ranks(mat: np.ndarray) -> np.ndarray:
    """Per-cell rank inside each target row (1=best), with min-rank tie handling."""
    rr = np.full(mat.shape, np.nan, dtype=float)
    for i in range(mat.shape[0]):
        row = mat[i, :]
        if np.all(np.isnan(row)):
            continue
        valid_vals = row[~np.isnan(row)]
        for j in range(mat.shape[1]):
            v = row[j]
            if np.isnan(v):
                continue
            rr[i, j] = 1 + np.sum(valid_vals > v)
    return rr


def _write_diag_table(path: Path, targets: List[str], n_levels: np.ndarray, ranks: np.ndarray, gaps: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["target_group", "n_levels", "diagonal_rank", "best_minus_diagonal_solved_rate"],
        )
        w.writeheader()
        for i, tg in enumerate(targets):
            w.writerow(
                {
                    "target_group": tg,
                    "n_levels": int(n_levels[i]),
                    "diagonal_rank": "" if np.isnan(ranks[i]) else int(ranks[i]),
                    "best_minus_diagonal_solved_rate": "" if np.isnan(gaps[i]) else f"{gaps[i]:.6f}",
                }
            )


def _plot(
    mat: np.ndarray,
    targets: List[str],
    models: List[str],
    n_levels: np.ndarray,
    ranks: np.ndarray,
    gaps: np.ndarray,
    row_ranks: np.ndarray,
    out_png: Path,
) -> None:
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.7], wspace=0.24)

    ax_h = fig.add_subplot(gs[0, 0])
    im = ax_h.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_h.set_title("Cross-Evaluation Performance Matrix Across Static Clusters")
    ax_h.set_xlabel("Source model cluster")
    ax_h.set_ylabel("Target evaluation cluster")
    ax_h.set_xticks(np.arange(len(models)))
    ax_h.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax_h.set_yticks(np.arange(len(targets)))
    ax_h.set_yticklabels(targets, fontsize=9)

    model_to_idx = {m: i for i, m in enumerate(models)}
    for i, tg in enumerate(targets):
        j = model_to_idx.get(tg)
        if j is None:
            continue
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False, edgecolor="white", linewidth=1.4)
        ax_h.add_patch(rect)

    # Show per-cell rank in the lower-right corner (1=best model for this target row).
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(row_ranks[i, j]):
                continue
            txt = str(int(row_ranks[i, j]))
            color = "white" if mat[i, j] < 0.50 else "black"
            ax_h.text(
                j + 0.44,
                i + 0.44,
                txt,
                ha="right",
                va="bottom",
                fontsize=6.2,
                color=color,
                alpha=0.95,
            )

    cbar = fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.02)
    cbar.set_label("Success rate; cell text = within-row rank")

    ax_r = fig.add_subplot(gs[0, 1], sharey=ax_h)
    y = np.arange(len(targets))
    max_rank = max(1, len(models))
    valid = ~np.isnan(ranks)
    sizes = np.where(n_levels > 0, 80 + 320 * (n_levels / np.max(n_levels)), 80)
    sc = ax_r.scatter(
        ranks[valid],
        y[valid],
        s=sizes[valid],
        c=gaps[valid],
        cmap="coolwarm",
        vmin=-0.05,
        vmax=0.05,
        edgecolors="black",
        linewidths=0.5,
    )
    ax_r.axvline(1.0, linestyle="--", color="black", linewidth=1.0)
    ax_r.set_xlim(0.5, max_rank + 0.5)
    ax_r.set_xticks(np.arange(1, max_rank + 1))
    ax_r.set_xlabel("Rank of matched-cluster model")
    ax_r.set_title("Matched-Cluster Rank by Target Cluster")
    ax_r.grid(axis="x", alpha=0.25)
    ax_r.tick_params(axis="y", left=False, labelleft=False)

    cbar2 = fig.colorbar(sc, ax=ax_r, fraction=0.07, pad=0.06)
    cbar2.set_label("Performance gap to row optimum")

    # annotate each point with target id suffix for readability
    for i, tg in enumerate(targets):
        if np.isnan(ranks[i]):
            continue
        ax_r.text(ranks[i] + 0.08, i, tg.replace("group_", ""), va="center", fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot crosspack solved-rate heatmap with diagonal-rank side panel."
    )
    p.add_argument(
        "--raw_dir",
        default="results/evaluate/letslogic/static_clusters/crosspack/raw",
        help="Directory containing target_group_XXX/model_group_YYY.csv files.",
    )
    p.add_argument(
        "--out_dir",
        default="results/plots/letslogic_crosspack",
        help="Output directory for figure and summary CSV.",
    )
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat, targets, models, n_levels = _collect_matrix(raw_dir)
    ranks, gaps = _diagonal_ranks(mat, targets, models)
    row_ranks = _row_ranks(mat)

    _plot(
        mat=mat,
        targets=targets,
        models=models,
        n_levels=n_levels,
        ranks=ranks,
        gaps=gaps,
        row_ranks=row_ranks,
        out_png=out_dir / "crosspack_heatmap_with_diagonal_rank.png",
    )
    _write_diag_table(out_dir / "crosspack_diagonal_rank.csv", targets, n_levels, ranks, gaps)

    top1 = int(np.sum(ranks == 1))
    valid = int(np.sum(~np.isnan(ranks)))
    print(f"[OK] figure: {out_dir / 'crosspack_heatmap_with_diagonal_rank.png'}")
    print(f"[OK] table : {out_dir / 'crosspack_diagonal_rank.csv'}")
    print(f"[OK] diagonal top-1: {top1}/{valid}")


if __name__ == "__main__":
    main()
