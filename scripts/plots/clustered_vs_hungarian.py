from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Row:
    level_id: str
    success: bool
    nodes: int
    runtime_s: float


@dataclass(frozen=True)
class GroupMetrics:
    group_id: str
    n_levels: int
    n_all3_solved: int
    solved_rate_hungarian: float
    solved_rate_general: float
    solved_rate_clustered: float
    runtime_hungarian_all3: float
    runtime_general_all3: float
    runtime_clustered_all3: float
    nodes_hungarian_all3: float
    nodes_general_all3: float
    nodes_clustered_all3: float


def read_eval_csv(path: Path) -> Dict[str, Row]:
    rows: Dict[str, Row] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            level_id = (row.get("level_id") or "").strip()
            if not level_id:
                continue
            rows[level_id] = Row(
                level_id=level_id,
                success=str(row.get("success", "")).strip() == "True",
                nodes=int(float(row.get("nodes") or 0)),
                runtime_s=float(row.get("runtime") or 0.0),
            )
    return rows


def get_group_id(path: Path) -> str:
    if not path.stem.startswith("group_"):
        raise ValueError(f"Unexpected group file name: {path.name}")
    return path.stem


def find_clustered_csv(crosspack_dir: Path, group_id: str) -> Path:
    nested = crosspack_dir / "raw" / f"target_{group_id}" / f"model_{group_id}.csv"
    if nested.exists():
        return nested
    fallback = crosspack_dir / "raw" / f"model_{group_id}.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Clustered csv for {group_id} not found. Tried: {nested} and {fallback}"
    )


def safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.array(values, dtype=float)))


def aggregate_group(
    group_id: str,
    hun_rows: Dict[str, Row],
    gen_rows: Dict[str, Row],
    clu_rows: Dict[str, Row],
) -> GroupMetrics:
    common = sorted(set(hun_rows.keys()) & set(gen_rows.keys()) & set(clu_rows.keys()))
    if not common:
        raise ValueError(f"No common levels between three methods for {group_id}")

    sr_h = float(np.mean(np.array([hun_rows[l].success for l in common], dtype=float)))
    sr_g = float(np.mean(np.array([gen_rows[l].success for l in common], dtype=float)))
    sr_c = float(np.mean(np.array([clu_rows[l].success for l in common], dtype=float)))

    all3 = [
        lid
        for lid in common
        if hun_rows[lid].success and gen_rows[lid].success and clu_rows[lid].success
    ]

    rt_h = safe_mean([hun_rows[l].runtime_s for l in all3])
    rt_g = safe_mean([gen_rows[l].runtime_s for l in all3])
    rt_c = safe_mean([clu_rows[l].runtime_s for l in all3])

    nd_h = safe_mean([float(hun_rows[l].nodes) for l in all3])
    nd_g = safe_mean([float(gen_rows[l].nodes) for l in all3])
    nd_c = safe_mean([float(clu_rows[l].nodes) for l in all3])

    return GroupMetrics(
        group_id=group_id,
        n_levels=len(common),
        n_all3_solved=len(all3),
        solved_rate_hungarian=sr_h,
        solved_rate_general=sr_g,
        solved_rate_clustered=sr_c,
        runtime_hungarian_all3=rt_h,
        runtime_general_all3=rt_g,
        runtime_clustered_all3=rt_c,
        nodes_hungarian_all3=nd_h,
        nodes_general_all3=nd_g,
        nodes_clustered_all3=nd_c,
    )


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_solved_rate_per_group(groups: List[GroupMetrics], out_path: Path) -> None:
    labels = [g.group_id for g in groups]
    x = np.arange(len(labels))
    width = 0.26

    h = np.array([g.solved_rate_hungarian for g in groups], dtype=float)
    gen = np.array([g.solved_rate_general for g in groups], dtype=float)
    clu = np.array([g.solved_rate_clustered for g in groups], dtype=float)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x - width, h, width=width, label="Hungarian", color="#ff7f0e")
    ax.bar(x, gen, width=width, label="General GNN", color="#1f77b4")
    ax.bar(x + width, clu, width=width, label="Clustered GNN", color="#2ca02c")
    ax.set_title("Solved Rate per Group")
    ax.set_ylabel("Solved rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    save_plot(fig, out_path)


def plot_runtime_nodes_all3(groups: List[GroupMetrics], out_path: Path) -> None:
    labels = [g.group_id for g in groups]
    x = np.arange(len(labels))
    width = 0.26

    rt_h = np.array([g.runtime_hungarian_all3 for g in groups], dtype=float)
    rt_g = np.array([g.runtime_general_all3 for g in groups], dtype=float)
    rt_c = np.array([g.runtime_clustered_all3 for g in groups], dtype=float)

    nd_h = np.array([g.nodes_hungarian_all3 for g in groups], dtype=float)
    nd_g = np.array([g.nodes_general_all3 for g in groups], dtype=float)
    nd_c = np.array([g.nodes_clustered_all3 for g in groups], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9.5), sharex=True)

    ax1.bar(x - width, rt_h, width=width, label="Hungarian", color="#ff7f0e")
    ax1.bar(x, rt_g, width=width, label="General GNN", color="#1f77b4")
    ax1.bar(x + width, rt_c, width=width, label="Clustered GNN", color="#2ca02c")
    ax1.set_title("Runtime on Levels Solved by All Three Methods")
    ax1.set_ylabel("Runtime (s)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=True)

    ax2.bar(x - width, nd_h, width=width, label="Hungarian", color="#ff7f0e")
    ax2.bar(x, nd_g, width=width, label="General GNN", color="#1f77b4")
    ax2.bar(x + width, nd_c, width=width, label="Clustered GNN", color="#2ca02c")
    ax2.set_title("Nodes on Levels Solved by All Three Methods")
    ax2.set_ylabel("Expanded nodes")
    ax2.set_xticks(x, labels, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.25)

    for i, g in enumerate(groups):
        ax2.text(i, 0.0, f"n={g.n_all3_solved}", fontsize=8, ha="center", va="bottom", rotation=90)

    save_plot(fig, out_path)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def plot_macro_micro_panel(groups: List[GroupMetrics], out_path: Path) -> None:
    n_levels = np.array([g.n_levels for g in groups], dtype=float)
    n_all3 = np.array([g.n_all3_solved for g in groups], dtype=float)

    solved = {
        "Hungarian": np.array([g.solved_rate_hungarian for g in groups], dtype=float),
        "General GNN": np.array([g.solved_rate_general for g in groups], dtype=float),
        "Clustered GNN": np.array([g.solved_rate_clustered for g in groups], dtype=float),
    }
    runtime = {
        "Hungarian": np.array([g.runtime_hungarian_all3 for g in groups], dtype=float),
        "General GNN": np.array([g.runtime_general_all3 for g in groups], dtype=float),
        "Clustered GNN": np.array([g.runtime_clustered_all3 for g in groups], dtype=float),
    }
    nodes = {
        "Hungarian": np.array([g.nodes_hungarian_all3 for g in groups], dtype=float),
        "General GNN": np.array([g.nodes_general_all3 for g in groups], dtype=float),
        "Clustered GNN": np.array([g.nodes_clustered_all3 for g in groups], dtype=float),
    }

    methods = ["Hungarian", "General GNN", "Clustered GNN"]
    colors = {"Hungarian": "#ff7f0e", "General GNN": "#1f77b4", "Clustered GNN": "#2ca02c"}

    solved_macro = [float(np.nanmean(solved[m])) for m in methods]
    solved_micro = [weighted_mean(solved[m], n_levels) for m in methods]
    runtime_macro = [float(np.nanmean(runtime[m])) for m in methods]
    runtime_micro = [weighted_mean(runtime[m], n_all3) for m in methods]
    nodes_macro = [float(np.nanmean(nodes[m])) for m in methods]
    nodes_micro = [weighted_mean(nodes[m], n_all3) for m in methods]

    x = np.array([0, 1], dtype=float)
    width = 0.23
    offsets = [-width, 0.0, width]
    labels = ["Macro", "Micro"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    ax_sr, ax_rt, ax_nd, ax_txt = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for i, m in enumerate(methods):
        ax_sr.bar(x + offsets[i], [solved_macro[i], solved_micro[i]], width=width, color=colors[m], label=m)
    ax_sr.set_title("Solved Rate: Macro vs Micro")
    ax_sr.set_xticks(x, labels)
    ax_sr.set_ylim(0.0, 1.05)
    ax_sr.grid(axis="y", alpha=0.25)
    ax_sr.legend(frameon=True)

    for i, m in enumerate(methods):
        ax_rt.bar(x + offsets[i], [runtime_macro[i], runtime_micro[i]], width=width, color=colors[m], label=m)
    ax_rt.set_title("Runtime (all-3-solved): Macro vs Micro")
    ax_rt.set_xticks(x, labels)
    ax_rt.set_ylabel("seconds")
    ax_rt.grid(axis="y", alpha=0.25)

    for i, m in enumerate(methods):
        ax_nd.bar(x + offsets[i], [nodes_macro[i], nodes_micro[i]], width=width, color=colors[m], label=m)
    ax_nd.set_title("Nodes (all-3-solved): Macro vs Micro")
    ax_nd.set_xticks(x, labels)
    ax_nd.set_ylabel("nodes")
    ax_nd.grid(axis="y", alpha=0.25)

    ax_txt.axis("off")
    micro_delta_sr = solved_micro[2] - solved_micro[1]
    macro_delta_sr = solved_macro[2] - solved_macro[1]
    summary_text = (
        f"Groups: {len(groups)}\n"
        f"Total levels (micro weights): {int(np.sum(n_levels))}\n"
        f"Total all-3-solved levels: {int(np.sum(n_all3))}\n\n"
        f"Clustered - General solved rate:\n"
        f"  Macro: {macro_delta_sr:+.4f}\n"
        f"  Micro: {micro_delta_sr:+.4f}\n"
    )
    ax_txt.text(
        0.02,
        0.98,
        summary_text,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    ax_txt.set_title("Summary")

    save_plot(fig, out_path)


def write_summary_csv(groups: List[GroupMetrics], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "group_id",
            "n_levels",
            "solved_rate_hungarian",
            "solved_rate_general_gnn",
            "solved_rate_clustered_gnn",
            "n_all3_solved",
            "runtime_hungarian_all3",
            "runtime_general_gnn_all3",
            "runtime_clustered_gnn_all3",
            "nodes_hungarian_all3",
            "nodes_general_gnn_all3",
            "nodes_clustered_gnn_all3",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for g in groups:
            writer.writerow(
                {
                    "group_id": g.group_id,
                    "n_levels": g.n_levels,
                    "solved_rate_hungarian": f"{g.solved_rate_hungarian:.6f}",
                    "solved_rate_general_gnn": f"{g.solved_rate_general:.6f}",
                    "solved_rate_clustered_gnn": f"{g.solved_rate_clustered:.6f}",
                    "n_all3_solved": g.n_all3_solved,
                    "runtime_hungarian_all3": f"{g.runtime_hungarian_all3:.6f}",
                    "runtime_general_gnn_all3": f"{g.runtime_general_all3:.6f}",
                    "runtime_clustered_gnn_all3": f"{g.runtime_clustered_all3:.6f}",
                    "nodes_hungarian_all3": f"{g.nodes_hungarian_all3:.6f}",
                    "nodes_general_gnn_all3": f"{g.nodes_general_all3:.6f}",
                    "nodes_clustered_gnn_all3": f"{g.nodes_clustered_all3:.6f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot clustered-vs-general-vs-hungarian evaluation charts for static clusters."
    )
    parser.add_argument(
        "--root_dir",
        default="results/evaluate/letslogic/static_clusters",
        help="Root directory with {crosspack,general} subdirs.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/plots/clusters",
        help="Directory to save produced plots and summary csv.",
    )
    args = parser.parse_args()

    root = Path(args.root_dir)
    hun_dir = root / "general" / "hungarian"
    gen_dir = root / "general" / "gnn"
    crosspack_dir = root / "crosspack"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hun_group_files = sorted(hun_dir.glob("group_*.csv"))
    if not hun_group_files:
        raise SystemExit(f"No group files found in {hun_dir}")

    groups: List[GroupMetrics] = []
    for hun_csv in hun_group_files:
        group_id = get_group_id(hun_csv)
        gen_csv = gen_dir / hun_csv.name
        if not gen_csv.exists():
            raise FileNotFoundError(f"Missing general gnn csv: {gen_csv}")
        clu_csv = find_clustered_csv(crosspack_dir, group_id)

        groups.append(
            aggregate_group(
                group_id=group_id,
                hun_rows=read_eval_csv(hun_csv),
                gen_rows=read_eval_csv(gen_csv),
                clu_rows=read_eval_csv(clu_csv),
            )
        )

    groups.sort(key=lambda g: g.group_id)

    plot_solved_rate_per_group(groups, out_dir / "01_solved_rate_per_group.png")
    plot_runtime_nodes_all3(groups, out_dir / "02_runtime_nodes_all3_solved.png")
    plot_macro_micro_panel(groups, out_dir / "03_macro_vs_micro_summary_panel.png")
    write_summary_csv(groups, out_dir / "summary_per_group.csv")

    print(f"[DONE] plots + summary saved to: {out_dir}")


if __name__ == "__main__":
    main()
