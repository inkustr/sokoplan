from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _domain_label(name: str) -> str:
    if "boxoban" in name.lower():
        return "Boxoban"
    if "cluster" in name.lower() or "group001" in name.lower() or "group_001" in name.lower() or "static_group001" in name.lower():
        return "Cluster"
    return name


def _load_summary(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "domains" not in data:
        raise SystemExit(f"Invalid summary JSON (missing 'domains'): {path}")
    return data


def _extract_cost_rows(summary: Dict[str, object]) -> List[Tuple[str, str, Dict[str, float]]]:
    rows: List[Tuple[str, str, Dict[str, float]]] = []
    domains = summary.get("domains", {})
    if not isinstance(domains, dict):
        return rows

    for dname, dobj in domains.items():
        if not isinstance(dobj, dict):
            continue
        modes = dobj.get("modes", {})
        if not isinstance(modes, dict):
            continue
        for mode in ("speed", "optimal_mix"):
            m = modes.get(mode)
            if not isinstance(m, dict):
                continue
            rows.append((_domain_label(str(dname)), mode, m))
    return rows


def _plot_speed_optmix_cost(ax_time, ax_nodes, rows: List[Tuple[str, str, Dict[str, float]]]) -> None:
    domains = sorted({d for d, _, _ in rows})
    mode_order = ["speed", "optimal_mix"]
    width = 0.34
    x = np.arange(len(domains))

    colors = {"speed": "#1f77b4", "optimal_mix": "#ff7f0e"}

    for i, mode in enumerate(mode_order):
        tvals = []
        nvals = []
        for d in domains:
            rec = next((r for r in rows if r[0] == d and r[1] == mode), None)
            if rec is None:
                tvals.append(0.0)
                nvals.append(0.0)
            else:
                stats = rec[2]
                tvals.append(float(stats.get("total_time_s", 0.0)))
                nvals.append(float(stats.get("total_nodes", 0.0)))

        offs = x + (i - 0.5) * width
        ax_time.bar(offs, tvals, width=width, color=colors[mode], label=mode)
        ax_nodes.bar(offs, nvals, width=width, color=colors[mode], label=mode)

    ax_time.set_title("Total runtime by heuristic mode")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(domains)
    ax_time.set_ylabel("seconds")
    ax_time.grid(axis="y", alpha=0.25)
    ax_time.legend(frameon=True, fontsize=9)

    ax_nodes.set_title("Total expanded nodes by heuristic mode")
    ax_nodes.set_xticks(x)
    ax_nodes.set_xticklabels(domains)
    ax_nodes.set_ylabel("expanded nodes")
    ax_nodes.grid(axis="y", alpha=0.25)
    ax_nodes.ticklabel_format(style="plain", axis="y")
    ax_nodes.legend(frameon=True, fontsize=9)


def _plot_fallback_composition(ax, rows: List[Tuple[str, str, Dict[str, float]]]) -> None:
    # x-axis = domain x mode (4 bars), stacked by relation.
    bars = []
    for d, mode, stats in rows:
        bars.append(
            {
                "label": f"{d}\n{mode}",
                "h_lt_g": float(stats.get("hungarian_lt_gnn", 0.0)),
                "g_lt_h": float(stats.get("gnn_lt_hungarian", 0.0)),
                "eq": float(stats.get("hungarian_eq_gnn", 0.0)),
            }
        )

    # Keep stable order: Boxoban speed,optmix then cluster speed,optmix
    def _k(b: Dict[str, float]) -> Tuple[int, int]:
        label = str(b["label"])
        is_group = 1 if "cluster" in label else 0
        is_opt = 1 if "optimal_mix" in label else 0
        return (is_group, is_opt)

    bars.sort(key=_k)
    labels = [b["label"] for b in bars]
    x = np.arange(len(labels))

    h_lt_g = np.array([b["h_lt_g"] for b in bars], dtype=float)
    g_lt_h = np.array([b["g_lt_h"] for b in bars], dtype=float)
    eq = np.array([b["eq"] for b in bars], dtype=float)
    total = np.maximum(h_lt_g + g_lt_h + eq, 1.0)

    # Plot as percentages for comparability across domains.
    p_hlt = 100.0 * h_lt_g / total
    p_glt = 100.0 * g_lt_h / total
    p_eq = 100.0 * eq / total

    ax.bar(x, p_hlt, color="#2ca02c", label="Hungarian < GNN")
    ax.bar(x, p_glt, bottom=p_hlt, color="#d62728", label="GNN < Hungarian")
    ax.bar(x, p_eq, bottom=p_hlt + p_glt, color="#7f7f7f", label="equal")

    ax.set_title("Fallback rate of optimal_mix")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("share of comparable states (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True, fontsize=9, loc="lower right")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot speed vs optimal_mix summary charts.")
    ap.add_argument("--summary_json", default="results/search_modes/summary.json")
    ap.add_argument("--out", default="results/search_modes/search_modes_plots.png")
    args = ap.parse_args()

    summary_path = Path(args.summary_json)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(summary_path)
    rows = _extract_cost_rows(summary)
    if not rows:
        raise SystemExit(f"No domain/mode rows parsed from: {summary_path}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    _plot_speed_optmix_cost(axes[0], axes[1], rows)
    _plot_fallback_composition(axes[2], rows)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[DONE] wrote plot: {out_path}")


if __name__ == "__main__":
    main()
