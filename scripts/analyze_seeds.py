#!/usr/bin/env python3
"""
Minimal multi-seed analysis of episodes_all_seeds_unified.csv.

Designed to be lean — no redundant plots, no per-city / per-scenario fan-out
that produces 50+ PNGs. The four figures below are sufficient for the
publication tables; everything else can be sliced in Excel from the CSVs.

Outputs (under results/Analysis_Seeds/):

  CSVs
  ----
  summary_by_method.csv
      One row per method. For each metric: mean across all observations,
      std across seed-means (i.e. how much the method's mean shifts between
      seeds — the "stability" signal), and std across all observations
      (pooled spread).

  summary_by_method_by_seed.csv
      One row per (method, seed). Mean of each metric within that subset —
      lets the reader see exactly how each seed shifted the headline number.

  summary_by_method_by_scenario.csv
      One row per (method, scenario), means aggregated across seeds and cities.

  Plots
  -----
  fig01_bars_means_seed_errors.png
      6 subplots (one metric each): bar = mean across 150 observations,
      error bar = std across the 3 per-seed means. Y-scale per-metric so
      throughput (0-1) and compute (ms-to-s) don't fight for the same axis.

  fig02_box_seed_variability.png
      6 subplots: box plot per method, colored by seed. Shows the FULL
      distribution per seed → reveals robustness vs sensitivity.

  fig03_throughput_by_scenario.png
      Grouped bars: throughput per (method, scenario), means across seeds
      and cities. Tests regime sensitivity in ONE figure (no per-city fan-out).

  fig04_pareto_throughput_vs_latency.png
      One scatter: x = mean latency (lower = better), y = mean throughput
      (higher = better). Methods in the top-left dominate. Error bars from
      seed std on both axes.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Methods listed in a stable order so all figures use the same color mapping.
METHOD_ORDER = [
    "MAPPO",
    "IPPO",
    "FaithfulMAPPER",
    "Hybrid",
    "TokenPassing",
    "HybridAdaptivePredictive",
]
METHOD_LABEL = {
    "MAPPO":                    "MAPPO",
    "IPPO":                     "IPPO",
    "FaithfulMAPPER":           "MAPPER",
    "Hybrid":                   "Hybrid",
    "TokenPassing":             "TP",
    "HybridAdaptivePredictive": "HAPC",
}

# Metrics selected for the focused 6-panel figures. Each is a publication
# headline; everything else is derivable from the unified CSV directly.
METRICS: list[tuple[str, str, bool]] = [
    # (column,                      display label,                    log_y)
    ("throughput_rate",             "Throughput rate",                False),
    ("latency_mean",                "Mean latency (steps)",           False),
    ("mean_bpr_along_route",        "Mean BPR along route",           False),
    ("agent_completed_gini",        "Agent-load Gini (fairness)",     False),
    ("delivery_route_efficiency",   "Delivery route efficiency",      False),
    ("compute_time_per_task_ms",    "Compute per task (ms)",          True),
]


def _setup_method_palette(df: pd.DataFrame) -> dict[str, str]:
    """Stable color per method based on METHOD_ORDER (extras get tab10 cycle)."""
    cmap = plt.get_cmap("tab10")
    methods_present = [m for m in METHOD_ORDER if m in df["method"].unique()]
    extras = [m for m in df["method"].unique() if m not in METHOD_ORDER]
    palette: dict[str, str] = {}
    for i, m in enumerate(methods_present + extras):
        palette[m] = cmap(i % 10)
    return palette


def write_summaries(df: pd.DataFrame, out_dir: Path) -> None:
    # 1. Per-method summary.
    rows = []
    for m, sub in df.groupby("method"):
        row: dict[str, float | str] = {"method": m, "n_obs": len(sub)}
        for col, _label, _log in METRICS:
            vals = pd.to_numeric(sub[col], errors="coerce")
            seed_means = (
                sub.assign(_v=vals).groupby("seed")["_v"].mean()
            )
            row[f"{col}__mean"]       = float(vals.mean())
            row[f"{col}__std_pooled"] = float(vals.std(ddof=1))
            row[f"{col}__std_seed"]   = float(seed_means.std(ddof=1))
        rows.append(row)
    (
        pd.DataFrame(rows)
        .sort_values("method")
        .to_csv(out_dir / "summary_by_method.csv", index=False)
    )

    # 2. Per-(method, seed) summary.
    g = df.groupby(["method", "seed"])
    rows2 = []
    for (m, sd), sub in g:
        row = {"method": m, "seed": int(sd), "n_obs": len(sub)}
        for col, _label, _log in METRICS:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[col] = float(vals.mean())
        rows2.append(row)
    (
        pd.DataFrame(rows2)
        .sort_values(["method", "seed"])
        .to_csv(out_dir / "summary_by_method_by_seed.csv", index=False)
    )

    # 3. Per-(method, scenario) summary.
    g3 = df.groupby(["method", "scenario"])
    rows3 = []
    for (m, sc), sub in g3:
        row = {"method": m, "scenario": sc, "n_obs": len(sub)}
        for col, _label, _log in METRICS:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[col] = float(vals.mean())
        rows3.append(row)
    (
        pd.DataFrame(rows3)
        .sort_values(["scenario", "method"])
        .to_csv(out_dir / "summary_by_method_by_scenario.csv", index=False)
    )


def fig01_bars_means_seed_errors(df: pd.DataFrame, out_dir: Path, palette: dict[str, str]) -> None:
    """6 panels: bar = method mean, error bar = std across 3 per-seed means."""
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    n_methods = len(methods)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, label, log_y) in zip(axes.flatten(), METRICS):
        means = []
        errs  = []
        for m in methods:
            sub = df[df["method"] == m]
            seed_means = (
                sub.groupby("seed")[col]
                .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
            )
            means.append(seed_means.mean())
            # std across seeds — 0 if only one seed present.
            errs.append(seed_means.std(ddof=1) if len(seed_means) > 1 else 0.0)

        xs = np.arange(n_methods)
        ax.bar(
            xs, means, yerr=errs, capsize=4,
            color=[palette[m] for m in methods],
            edgecolor="black", linewidth=0.5,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=20, ha="right")
        ax.set_title(label)
        if log_y:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Means per method (error bars = std across seeds)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig01_bars_means_seed_errors.png", dpi=150)
    plt.close(fig)


def fig02_box_seed_variability(df: pd.DataFrame, out_dir: Path) -> None:
    """6 panels: box plots per (method, seed) so seed spread is visible."""
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    seeds   = sorted(df["seed"].unique())
    n_seeds = len(seeds)
    seed_cmap = plt.get_cmap("viridis", max(2, n_seeds))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (col, label, log_y) in zip(axes.flatten(), METRICS):
        # Layout: groups of (n_seeds) boxes per method, with gaps between methods.
        positions = []
        data       = []
        colors     = []
        box_width  = 0.6
        method_gap = 1.5
        cursor     = 0.0
        for m in methods:
            for k, sd in enumerate(seeds):
                sub = df[(df["method"] == m) & (df["seed"] == sd)]
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                positions.append(cursor + k * box_width)
                data.append(vals.values if len(vals) else [0.0])
                colors.append(seed_cmap(k))
            cursor += n_seeds * box_width + method_gap

        bp = ax.boxplot(
            data, positions=positions, widths=box_width * 0.85,
            patch_artist=True, showfliers=False,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
            patch.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black")

        # Method labels centered under each group.
        group_centers = [
            (n_seeds * box_width / 2 - box_width / 2)
            + i * (n_seeds * box_width + method_gap)
            for i in range(len(methods))
        ]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=20, ha="right")
        ax.set_title(label)
        if log_y:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

    # Single legend (one entry per seed) at the top of the figure.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=seed_cmap(k), alpha=0.65, ec="black")
        for k in range(n_seeds)
    ]
    legend_labels = [f"seed {s}" for s in seeds]
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", ncol=n_seeds, frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(
        "Per-method distribution by seed (box = quartiles, line = median)",
        fontsize=13, y=0.93,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_dir / "fig02_box_seed_variability.png", dpi=150)
    plt.close(fig)


def fig03_throughput_by_scenario(df: pd.DataFrame, out_dir: Path, palette: dict[str, str]) -> None:
    """Grouped bars: throughput per (method, scenario), averaged across seeds and cities."""
    methods   = [m for m in METHOD_ORDER if m in df["method"].unique()]
    scenarios = sorted(df["scenario"].unique())
    n_methods = len(methods)
    n_scen    = len(scenarios)
    bar_w     = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x_base = np.arange(n_scen)
    for i, m in enumerate(methods):
        means = []
        errs  = []
        for sc in scenarios:
            sub = df[(df["method"] == m) & (df["scenario"] == sc)]
            seed_means = sub.groupby("seed")["throughput_rate"].mean()
            means.append(seed_means.mean())
            errs.append(seed_means.std(ddof=1) if len(seed_means) > 1 else 0.0)
        ax.bar(
            x_base + (i - n_methods / 2 + 0.5) * bar_w,
            means, width=bar_w,
            yerr=errs, capsize=3,
            label=METHOD_LABEL[m], color=palette[m],
            edgecolor="black", linewidth=0.4,
        )
    ax.set_xticks(x_base)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.set_ylabel("Throughput rate")
    ax.set_title("Throughput per method × scenario (error bars = std across seeds)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig03_throughput_by_scenario.png", dpi=150)
    plt.close(fig)


def fig04_pareto(df: pd.DataFrame, out_dir: Path, palette: dict[str, str]) -> None:
    """Scatter: mean latency (x) vs mean throughput (y); error bars from seed std.

    Lower latency + higher throughput → top-left quadrant dominates.
    """
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for m in methods:
        sub = df[df["method"] == m]
        seed_means = sub.groupby("seed").agg(
            throughput=("throughput_rate", "mean"),
            latency=("latency_mean", "mean"),
        )
        x = seed_means["latency"].mean()
        y = seed_means["throughput"].mean()
        xe = seed_means["latency"].std(ddof=1) if len(seed_means) > 1 else 0.0
        ye = seed_means["throughput"].std(ddof=1) if len(seed_means) > 1 else 0.0
        ax.errorbar(
            x, y, xerr=xe, yerr=ye,
            fmt="o", markersize=9, color=palette[m],
            ecolor="gray", capsize=3, label=METHOD_LABEL[m],
            markeredgecolor="black", markeredgewidth=0.6,
        )
        ax.annotate(
            METHOD_LABEL[m], (x, y),
            xytext=(6, 6), textcoords="offset points", fontsize=9,
        )
    ax.set_xlabel("Mean latency (steps) — lower is better")
    ax.set_ylabel("Mean throughput rate — higher is better")
    ax.set_title("Throughput vs. latency (mean across seeds, error bars = seed std)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig04_pareto_throughput_vs_latency.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-file",
        default=r"c:\ConflictualMAS\results\Evaluation Files\episodes_all_seeds_unified.csv",
    )
    ap.add_argument(
        "--out-dir",
        default=r"c:\ConflictualMAS\results\Analysis_Seeds",
    )
    args = ap.parse_args()

    in_path = Path(args.in_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    # Numeric coercion for the metric columns (defensive — some legacy rows
    # may contain empty strings from very old CSV exports).
    for col, _label, _log in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Scenario name normalisation. Seed 42's unified file mixes the short
    # ("shock", "buildup") and full ("stress_shock", "stress_buildup") names
    # because that seed's HAPC and RL CSVs were exported from runs that used
    # different label conventions. Seeds 43/44 are consistent (full names).
    # Coalesce so per-scenario analysis treats them as the same regime.
    df["scenario"] = df["scenario"].replace({
        "shock":   "stress_shock",
        "buildup": "stress_buildup",
    })

    palette = _setup_method_palette(df)
    write_summaries(df, out_dir)
    fig01_bars_means_seed_errors(df, out_dir, palette)
    fig02_box_seed_variability(df, out_dir)
    fig03_throughput_by_scenario(df, out_dir, palette)
    fig04_pareto(df, out_dir, palette)

    print(f"[analyze] wrote 3 CSVs and 4 PNGs into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())