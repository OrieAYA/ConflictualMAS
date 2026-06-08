"""
Simple planning analysis.

Reads results/planning_test/planning_summary.csv and produces a focused
set of graphs on three axes:

    1. Throughput          — throughput_rate (higher = better)
    2. Computing time      — wallclock_ms, compute_time_per_task_ms (lower)
    3. Distance travelled  — total_dist_m (lower = better)

Output: results/planning_test/Analysis/

Usage:
    py results\\_analyze_planning.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import warnings

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"[FATAL] missing dependency: {e}")
    print("  py -m pip install pandas matplotlib numpy")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
CSV  = Path("C:/ConflictualMAS/results/planning_test/planning_summary.csv")
OUT  = Path("C:/ConflictualMAS/results/planning_test/Analysis")
OUT.mkdir(parents=True, exist_ok=True)

PLANNER_COLORS = {
    "MCA_cheapest_insertion":     "#1f77b4",
    "DbVNS_lifelong_replan":      "#ff7f0e",
    "DbVNS_hybrid_K3":            "#2ca02c",
    "ALNS_RopkePisinger":         "#d62728",
    "DoubleHorizon_MitrovicMinic":"#9467bd",
}
def color(p): return PLANNER_COLORS.get(p, "#999999")

# Order scenarios from light to heavy load.
SCENARIO_ORDER = ["low_saturation", "medium_saturation", "high_saturation"]


# ────────────────────────────────────────────────────────────────────────────
def bar_by(df, value_col, label, direction, fname, group_col="scenario"):
    """Bar plot: one group per `group_col`, one bar per planner.
    Error bars = standard error of the mean."""
    sub = df[[group_col, "mode", value_col]].dropna()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[value_col])
    if sub.empty:
        return False

    if group_col == "scenario":
        groups = [g for g in SCENARIO_ORDER if g in sub[group_col].unique()]
    else:
        groups = sorted(sub[group_col].unique())
    planners = sorted(sub["mode"].unique())
    x = np.arange(len(groups))
    bw = 0.8 / max(1, len(planners))

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(groups) + 2), 5))
    for i, p in enumerate(planners):
        g = sub[sub["mode"] == p].groupby(group_col)[value_col]
        means = g.mean().reindex(groups)
        sems  = g.sem().reindex(groups).fillna(0)
        ax.bar(x + i * bw, means.values, bw,
               yerr=sems.values, capsize=2, label=p,
               color=color(p), edgecolor="black", linewidth=0.3)

    ax.set_xticks(x + bw * (len(planners) - 1) / 2)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel(label)
    arrow = " (higher = better)" if direction == "higher" else " (lower = better)"
    ax.set_title(f"{label}{arrow}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)
    return True


def heatmap(df, value_col, label, direction, fname):
    """Heatmap city × planner, mean over scenarios + episodes."""
    sub = df[["city", "mode", value_col]].dropna()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return False
    piv = sub.pivot_table(index="city", columns="mode", values=value_col, aggfunc="mean")
    if piv.empty:
        return False

    cmap = "viridis" if direction == "higher" else "viridis_r"
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(piv.columns) + 2),
                                     max(3, 0.6 * len(piv.index) + 1.5)))
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    arrow = " (higher = better)" if direction == "higher" else " (lower = better)"
    ax.set_title(f"{label}{arrow}")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if pd.isna(v): continue
            txt = f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if direction == "lower" else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)
    return True


def pareto(df, x_col, x_label, x_dir, y_col, y_label, y_dir, fname):
    """One point per (planner, scenario, city) at mean across episodes."""
    needed = ["mode", "scenario", "city", x_col, y_col]
    sub = df[needed].dropna()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty: return False
    agg = sub.groupby(["mode","scenario","city"]).agg(
        x=(x_col,"mean"), y=(y_col,"mean")).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"low_saturation":"o", "medium_saturation":"s", "high_saturation":"^"}
    for m, g in agg.groupby("mode"):
        for sc, gsc in g.groupby("scenario"):
            ax.scatter(gsc["x"], gsc["y"], s=80,
                        marker=markers.get(sc, "o"),
                        label=f"{m} / {sc}",
                        color=color(m),
                        edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xlabel(x_label + (" ↑" if x_dir == "higher" else " ↓"))
    ax.set_ylabel(y_label + (" ↑" if y_dir == "higher" else " ↓"))
    ax.set_title(f"Pareto: {x_label} vs {y_label}")
    # Compact legend: one entry per planner (drop the per-scenario duplication)
    handles, labels = ax.get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, l in zip(handles, labels):
        planner = l.split(" / ")[0]
        if planner in seen: continue
        seen.add(planner)
        uh.append(h); ul.append(planner)
    ax.legend(uh, ul, fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)
    return True


# ────────────────────────────────────────────────────────────────────────────
def main():
    if not CSV.exists():
        print(f"[FATAL] {CSV} not found")
        sys.exit(1)
    print(f"Loading {CSV}")
    df = pd.read_csv(CSV)
    print(f"  rows    : {len(df)}")
    print(f"  planners: {sorted(df['mode'].unique())}")
    print(f"  scenars : {sorted(df['scenario'].unique())}")
    print(f"  cities  : {sorted(df['city'].unique())}")
    print(f"  output  → {OUT}\n")

    # ── 3 axes : 1 bar plot par scenario + 1 heatmap par city × planner ──
    AXES = [
        # (column, label, direction, file_slug)
        ("throughput_rate",          "Throughput rate",            "higher", "throughput"),
        ("wallclock_ms",             "Wallclock time (ms)",        "lower",  "wallclock"),
        ("compute_time_per_task_ms", "Compute time per task (ms)", "lower",  "compute_per_task"),
        ("total_dist_m",             "Total distance travelled (m)","lower",  "distance"),
    ]
    print("── Axis plots ──")
    for col, label, direction, slug in AXES:
        ok_scen = bar_by(df, col, label, direction, f"{slug}__by_scenario.png", "scenario")
        ok_city = bar_by(df, col, label, direction, f"{slug}__by_city.png",     "city")
        ok_heat = heatmap(df, col, label, direction, f"{slug}__heatmap.png")
        tags = "".join(["S" if ok_scen else "-",
                        "C" if ok_city else "-",
                        "H" if ok_heat else "-"])
        print(f"  [{tags}] {col:<28} → {slug}__*.png")

    # ── Pareto trade-offs (3 axes pairwise) ──
    print("\n── Pareto trade-offs ──")
    pareto_specs = [
        ("throughput_rate", "Throughput rate",         "higher",
         "wallclock_ms",    "Wallclock time (ms)",     "lower",
         "pareto_throughput_vs_wallclock.png"),
        ("throughput_rate", "Throughput rate",         "higher",
         "total_dist_m",    "Total distance (m)",      "lower",
         "pareto_throughput_vs_distance.png"),
        ("total_dist_m",    "Total distance (m)",      "lower",
         "wallclock_ms",    "Wallclock time (ms)",     "lower",
         "pareto_distance_vs_wallclock.png"),
        ("throughput_rate",          "Throughput rate",            "higher",
         "compute_time_per_task_ms", "Compute time per task (ms)", "lower",
         "pareto_throughput_vs_compute_per_task.png"),
    ]
    for x, xl, xd, y, yl, yd, fn in pareto_specs:
        ok = pareto(df, x, xl, xd, y, yl, yd, fn)
        print(f"  [{'OK' if ok else 'skip'}] {x:<25} vs {y}")

    # ── Summary CSV ──
    print("\n── Summary table ──")
    cols = ["throughput_rate", "wallclock_ms", "compute_time_per_task_ms",
            "total_dist_m", "latency_mean"]
    cols = [c for c in cols if c in df.columns]
    summary = (df.groupby(["mode", "scenario", "city"])[cols]
                  .mean().round(4).reset_index()
                  .sort_values(["city", "scenario", "mode"])
                  .reset_index(drop=True))
    summary.to_csv(OUT / "summary_mode_scenario_city.csv", index=False)
    print(f"  saved → {OUT / 'summary_mode_scenario_city.csv'}")

    per_mode = df.groupby("mode")[cols].mean().round(4)
    per_mode.to_csv(OUT / "summary_per_planner.csv")
    print(f"  saved → {OUT / 'summary_per_planner.csv'}\n")
    print(per_mode.to_string())

    n_png = sum(1 for _ in OUT.glob("*.png"))
    n_csv = sum(1 for _ in OUT.glob("*.csv"))
    print(f"\nDone — {n_png} PNG + {n_csv} CSV in {OUT}")


if __name__ == "__main__":
    main()
