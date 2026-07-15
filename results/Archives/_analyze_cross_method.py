"""
Cross-method analysis of LGPDP comparison runs.

Loads four CSV sources (RL allocation, TP allocation, HAPC standalone, plus
the new Paris_Small / NewYork_Small extension) and produces a unified set of
plots and a summary table along three axes:

    1. CAPACITE        (capacity / throughput)
        throughput_rate, tasks_completed / tasks_appeared, accept_rate
    2. QUALITE         (quality of delivery)
        latency_mean, mean_wait_steps, delivery_route_efficiency,
        agent_completed_gini, mean_extra_steps_per_task
    3. ENVIRONNEMENT   (environmental impact)
        mean_congestion, congestion_variance, mean_bpr_along_route,
        n_traversals_in_jam, route_congestion_exposure,
        total_fleet_distance_m

Per-axis output:
    - one bar chart per metric (mean +/- stderr across episodes, grouped by
      (city, scenario), one bar per method)
    - one heatmap (city x method, mean across scenarios x episodes)

Cross-axis output:
    - Pareto-style scatter plots (e.g. throughput vs latency, throughput vs
      congestion) so you can see the trade-offs at a glance.

Designed to run in parallel with the actual Option O / Option G runs - it
only reads existing CSV files and never modifies them. Re-run as the runs
finish to refresh the graphs incrementally.

Usage:
    python _analyze_cross_method.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import warnings

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # headless backend so we can run on a server
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"[FATAL] missing dependency: {e}")
    print("  pip install pandas matplotlib numpy")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# Configuration: where to find each CSV
# ────────────────────────────────────────────────────────────────────────────
RESULTS = Path("C:/ConflictualMAS/results")

# Preferred source: the unified all-seeds file. Already contains
# (seed, source, method, city, scenario, episode, …) so we get full inter-seed
# coverage from a single read. When present, we use it instead of the per-seed
# legacy sources below.
UNIFIED_ALL_SEEDS = RESULTS / "Evaluation Files" / "episodes_all_seeds_unified.csv"

SOURCES = {
    # name → (csv_path, schema_kind)
    #   schema_kind ∈ {"training", "solver"}
    # Legacy per-seed sources, used as fallback when the unified file is absent.
    "RL":   (RESULTS / "Evaluation Files" / "episodes_seed42_RL.csv",   "training"),
    "TP":   (RESULTS / "Evaluation Files" / "episodes_seed42_TP.csv",   "training"),
    "HAPC": (RESULTS / "Evaluation Files" / "episodes_seed42_HAPC.csv", "solver"),
    # Future Option O / G extension runs (auto-joined when present):
    "ORL_gen":   (RESULTS / "Option_O_results" / "episodes_seed42.csv",                           "training"),
    "HAPC_gen":  (RESULTS / "Option_G_results" / "sota_standalone_O" / "sota_standalone_seed42.csv", "solver"),
}

OUT_DIR = RESULTS / "Analysis_Cross_Method"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# 5-axes output structure (numeric prefix → ordering in file explorer).
for sub in ("01_throughput", "02_quality", "03_impact", "04_wallclock",
            "05_seeds", "summary", "pareto"):
    (OUT_DIR / sub).mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# Schema unification
# ────────────────────────────────────────────────────────────────────────────
# The two loggers use different column names for the SAME concept. We map
# everything to a single canonical name so downstream code is uniform.
TRAINING_TO_CANON = {
    "policy_mode":            "method",
    "n_agents_max":           "n_agents",
    "peak_congestion":        "peak_load",
    "time_lost_to_congestion_steps": "time_lost_to_congestion",
}
SOLVER_TO_CANON = {
    "solver":                 "method",
    # solver CSV uses 'scenario' directly; training CSV uses 'phase'.
}

# Extract the SCENARIO label from the training-csv "phase" column.
# Training "phase" values look like:
#   eval_normal_wave, generalize_stress_buildup, generalize_normal_ramp, ...
# We strip the leading {eval,generalize,stress,train}_ to get just the scenario.
PHASE_PREFIXES = ("eval_", "generalize_", "stress_", "train_")


def _unify_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=TRAINING_TO_CANON)
    # derive 'scenario' from 'phase' (strip the eval_/generalize_ prefix)
    if "phase" in df.columns:
        def _strip(p):
            for pre in PHASE_PREFIXES:
                if isinstance(p, str) and p.startswith(pre):
                    return p[len(pre):]
            return p
        df["scenario"] = df["phase"].map(_strip)
    return df


def _unify_solver(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=SOLVER_TO_CANON)
    return df


def load_all() -> pd.DataFrame:
    # Preferred path: unified all-seeds CSV. Already canonical-schema and
    # already carries the `seed` column we need for inter-seed analysis.
    if UNIFIED_ALL_SEEDS.exists() and UNIFIED_ALL_SEEDS.stat().st_size > 0:
        try:
            df = pd.read_csv(UNIFIED_ALL_SEEDS)
            if not df.empty:
                print(f"  [load] unified all-seeds: {len(df):5d} rows  ←  {UNIFIED_ALL_SEEDS}")
                if "seed" not in df.columns:
                    df["seed"] = -1
                if "source" not in df.columns:
                    df["source"] = "unified"
                _add_derived_metrics(df)
                return df
        except Exception as e:
            print(f"  [warn] unified read failed ({type(e).__name__}: {e}); falling back to legacy per-seed sources")

    frames = []
    for name, (path, kind) in SOURCES.items():
        if not path.exists():
            print(f"  [skip] {name} — file not found: {path}")
            continue
        try:
            sz = path.stat().st_size
        except OSError:
            sz = 0
        if sz == 0:
            # Common when an Option O / G run has just started writing — the
            # CSV header is still in the stdio buffer, file on disk is empty.
            print(f"  [skip] {name} — file is empty (0 bytes — run may still be writing): {path}")
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"  [skip] {name} — pandas: no columns to parse: {path}")
            continue
        except Exception as e:
            print(f"  [skip] {name} — read error ({type(e).__name__}: {e}): {path}")
            continue
        if df.empty:
            print(f"  [skip] {name} — header only, 0 data rows: {path}")
            continue
        print(f"  [load] {name}: {len(df):5d} rows  ←  {path}")
        df = _unify_training(df) if kind == "training" else _unify_solver(df)
        df["source"] = name
        if "seed" not in df.columns:
            df["seed"] = 42  # legacy per-seed file always seed 42
        frames.append(df)
    if not frames:
        print("[FATAL] no CSV loaded.")
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True, sort=False)
    _add_derived_metrics(df)
    return df


def _add_derived_metrics(df: pd.DataFrame) -> None:
    """In-place: compute per-delivery normalised impact metrics so the
    Impact axis can be read fairly across methods with very different
    throughput. A method that delivers few tasks naturally has lower
    ABSOLUTE fleet distance and jam traversals — but the EFFICIENCY per
    delivered task is what reveals true environmental cost.
    """
    if "tasks_completed" not in df.columns:
        return
    completed = pd.to_numeric(df["tasks_completed"], errors="coerce").clip(lower=1)
    for src, dst in (
        ("total_fleet_distance_m",   "fleet_distance_per_delivery_m"),
        ("n_traversals_in_jam",      "jam_traversals_per_delivery"),
        ("route_congestion_exposure","congestion_exposure_per_delivery"),
    ):
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce") / completed


# ────────────────────────────────────────────────────────────────────────────
# Metric definitions per axis
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
# Five analysis axes (re-grouped for SIGSPATIAL paper structure)
# Each tuple: (canonical_column, display_name, "higher"|"lower"|"context")
# ────────────────────────────────────────────────────────────────────────────

# 1. Throughput — capacity to fulfill tasks
THROUGHPUT_METRICS = [
    ("throughput_rate",          "Throughput rate",                "higher"),
    ("tasks_completed",          "Tasks completed (count)",        "higher"),
    ("accept_rate",              "Accept rate",                    "higher"),
    ("value_throughput_rate",    "Value-weighted throughput (RL)", "higher"),
]

# 2. Quality — time profile of accepted tasks + distribution fairness
# Note on Gini/CV: fairness depends on tasks_completed. A method completing
# few tasks can have artificially low CV; we surface this in the analysis.
QUALITY_METRICS = [
    ("latency_mean",                "Mean latency arrival→delivery (steps)", "lower"),
    ("mean_wait_steps",             "Mean wait arrival→pickup (steps)",      "lower"),
    ("mean_trip_steps",             "Mean trip pickup→delivery (steps)",     "lower"),
    ("delivery_route_efficiency",   "Delivery route efficiency",             "higher"),
    ("mean_extra_steps_per_task",   "Extra steps per task (detour)",         "lower"),
    ("agent_completed_gini",        "Agent completion Gini (fairness, caveat: low Gini may reflect few deliveries)", "lower"),
    ("agent_completed_std",         "Agent completion CV (fairness, caveat: low CV may reflect few deliveries)",     "lower"),
    ("completion_per_accepted",     "Completion per accepted (reliability)",  "higher"),
]

# 3. Impact — environmental footprint of each method
# IMPORTANT: absolute totals (fleet_distance, jam traversals) are unfair to
# compare across methods because methods with fewer deliveries naturally
# produce less footprint. We add per-delivery normalised variants below to
# isolate true efficiency.
IMPACT_METRICS = [
    ("mean_congestion",                "Mean network congestion (BPR mean load)", "lower"),
    ("congestion_variance",            "Congestion variance",                      "lower"),
    ("peak_load",                      "Peak congestion load",                     "lower"),
    ("mean_bpr_along_route",           "Mean BPR along agent routes",              "lower"),
    ("n_traversals_in_jam",            "Edge traversals in jam (absolute)",        "context"),
    ("route_congestion_exposure",      "Route congestion exposure",                "lower"),
    ("total_fleet_distance_m",         "Total fleet distance (m, absolute)",       "context"),
    # ── Per-delivery normalised (fairness across methods with different #deliveries)
    ("fleet_distance_per_delivery_m",  "Fleet distance per delivered task (m)",    "lower"),
    ("jam_traversals_per_delivery",    "Jam traversals per delivered task",        "lower"),
    ("congestion_exposure_per_delivery", "Congestion exposure per delivered task", "lower"),
]

# 4. Wallclock — computational complexity
WALLCLOCK_METRICS = [
    ("wallclock_ms",                   "Wallclock per episode (ms)",               "lower"),
    ("compute_time_per_task_ms",       "Compute per completed task (ms, bias↑ for low-throughput)", "lower"),
    ("compute_time_per_decision_us",   "Compute per Bernoulli decision (us)",      "lower"),
    # ── Path/allocation breakdown — populated when the instrumented binary is used
    ("path_compute_time_ms",           "Path compute time per episode (ms, new)",  "lower"),
    ("mean_pure_alloc_time_ms",        "Pure TAM+policy time per offer (ms, new)", "lower"),
    ("mean_allocation_time_us",        "TAM+policy+path-cache per offer (us)",     "lower"),
]

# 5. Seeds — inter-seed dispersion (analysed separately in seed_*_plot fns)
SEED_HEADLINE = [
    ("throughput_rate",          "Throughput rate",        "higher"),
    ("latency_mean",             "Mean latency (steps)",   "lower"),
    ("mean_bpr_along_route",     "Mean BPR along route",   "lower"),
    ("agent_completed_gini",     "Fairness (Gini)",        "lower"),
]

# Mapping axis name → (subfolder, metric list) for the main loop
ALL_AXES = {
    "throughput":  ("01_throughput", THROUGHPUT_METRICS),
    "quality":     ("02_quality",    QUALITY_METRICS),
    "impact":      ("03_impact",     IMPACT_METRICS),
    "wallclock":   ("04_wallclock",  WALLCLOCK_METRICS),
}

# ────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ────────────────────────────────────────────────────────────────────────────
COLORS = {
    "MAPPO":                     "#1f77b4",
    "FaithfulMAPPER":            "#9467bd",
    "MAPPER":                    "#8c564b",
    "IPPO":                      "#2ca02c",
    "Hybrid":                    "#17becf",
    "TokenPassing":              "#d62728",
    "HybridAdaptivePredictive":  "#ff7f0e",
}

def _color_for(method: str) -> str:
    return COLORS.get(method, "#999999")


def _clean_numeric(df: pd.DataFrame, cols, metric: str) -> pd.DataFrame:
    sub = df[cols].dropna()
    if sub.empty:
        return sub
    sub = sub.copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    return sub.dropna(subset=[metric])


def _aggregate_seed_aware(sub: pd.DataFrame, metric: str, by: list[str]):
    """Two-level mean when multiple seeds: episodes → per-seed mean → cross-seed mean.
    Returns (means_df, errs_df, err_label) all keyed by `by`."""
    n_seeds = sub["seed"].nunique() if "seed" in sub.columns else 1
    if n_seeds > 1:
        per_seed = sub.groupby(["seed"] + by)[metric].mean().reset_index()
        agg = per_seed.groupby(by)[metric].agg(["mean", "std"]).reset_index()
        agg["err"] = agg["std"].fillna(0)
        return agg, f"±std across {n_seeds} seeds"
    agg = sub.groupby(by)[metric].agg(["mean", "sem"]).reset_index()
    agg["err"] = agg["sem"].fillna(0)
    return agg, "±sem across episodes"


# Wallclock-style metrics with very wide dynamic range — log scale on bars
# avoids crushing small values when the heaviest method is 100× slower.
_LOG_SCALE_METRICS = {
    "wallclock_ms",
    "compute_time_per_task_ms",
    "compute_time_per_decision_us",
    "mean_allocation_time_us",
    "path_compute_time_ms",
    "mean_pure_alloc_time_ms",
}


def bar_plot_metric(df: pd.DataFrame, metric: str, label: str, direction: str,
                    out_path: Path, group_by: str = "city"):
    """Bar chart: one group per `group_by` value, one bar per method.

    Aggregation when multiple seeds are present:
        episodes  → mean per (seed, method, group)   (one point per seed)
        seeds     → mean ± std across seeds          (the bar + error bar)
    With a single seed this degenerates to the previous behaviour
    (mean across episodes, SEM as error bar)."""
    if metric not in df.columns:
        return False
    cols = ["seed", group_by, "method", metric] if "seed" in df.columns \
        else [group_by, "method", metric]
    sub = _clean_numeric(df, cols, metric)
    if sub.empty:
        return False

    agg, err_label = _aggregate_seed_aware(sub, metric, ["method", group_by])
    groups  = sorted(sub[group_by].unique())
    methods = sorted(sub["method"].unique())
    x = np.arange(len(groups))
    bw = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(groups)), 5))
    for i, m in enumerate(methods):
        msub = agg[agg["method"] == m].set_index(group_by)
        means = msub["mean"].reindex(groups)
        errs  = msub["err"].reindex(groups).fillna(0)
        ax.bar(x + i * bw, means.values, bw,
               yerr=errs.values, capsize=2,
               label=m, color=_color_for(m), edgecolor="black", linewidth=0.3)

    ax.set_xticks(x + bw * (len(methods) - 1) / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel(label)
    arrow = "  (higher = better)" if direction == "higher" else \
            ("  (lower = better)" if direction == "lower" else "")
    log_suffix = ""
    if metric in _LOG_SCALE_METRICS:
        ax.set_yscale("log")
        log_suffix = "  [log scale]"
    ax.set_title(f"{label}{arrow}{log_suffix}\n(error bars: {err_label})")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def heatmap_metric(df: pd.DataFrame, metric: str, label: str, direction: str,
                    out_path: Path):
    """Heatmap: city × method, mean over scenarios + episodes + seeds.

    With multiple seeds we first collapse each seed to its own mean (so all
    seeds get equal weight regardless of how many episodes they ran), then
    average those per-seed means."""
    if metric not in df.columns:
        return False
    cols = ["seed", "city", "method", metric] if "seed" in df.columns \
        else ["city", "method", metric]
    sub = _clean_numeric(df, cols, metric)
    if sub.empty:
        return False
    if "seed" in sub.columns and sub["seed"].nunique() > 1:
        per_seed = sub.groupby(["seed", "city", "method"])[metric].mean().reset_index()
        piv = per_seed.pivot_table(index="city", columns="method",
                                   values=metric, aggfunc="mean")
    else:
        piv = sub.pivot_table(index="city", columns="method",
                              values=metric, aggfunc="mean")
    if piv.empty:
        return False

    # Sequential blue colormap: darker = larger magnitude. Combined with the
    # title "(lower / higher = better)" the reader still understands direction
    # while avoiding the bright-yellow end of viridis (poor print readability).
    cmap = "Blues"
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(piv.columns) + 2),
                                     max(3, 0.5 * len(piv.index) + 1.5)))
    # Wide-dynamic-range wallclock metrics: log-scale the color norm so the
    # difference between 50ms and 8000ms becomes visible.
    norm = None
    if metric in _LOG_SCALE_METRICS:
        import matplotlib.colors as mcolors
        # Guard against zeros/negatives that would break LogNorm.
        positive = piv.values[piv.values > 0]
        if positive.size > 0:
            vmin = float(positive.min())
            vmax = float(piv.values.max())
            if vmax > vmin > 0:
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    arrow = "  (higher = better)" if direction == "higher" else \
            ("  (lower = better)" if direction == "lower" else "")
    ax.set_title(f"{label}{arrow}")
    # Annotate cells. With sequential Blues colormap, dark cells (large
    # values) need white text; light cells (small values) need black.
    # Threshold at the colormap midpoint via the normalised data value.
    arr = piv.values
    if metric in _LOG_SCALE_METRICS and norm is not None:
        # In log space, "dark" begins around the geometric mean of vmin/vmax.
        threshold = (norm.vmin * norm.vmax) ** 0.5
    else:
        finite = arr[~pd.isna(arr)]
        threshold = (finite.min() + finite.max()) / 2 if finite.size > 0 else 0
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = arr[i, j]
            if pd.isna(v):
                continue
            txt = f"{v:.2f}" if abs(v) < 1000 else f"{v:.0f}"
            text_color = "white" if v >= threshold else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=text_color)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def pareto_plot(df: pd.DataFrame, x_metric: str, x_label: str, x_dir: str,
                 y_metric: str, y_label: str, y_dir: str, out_path: Path):
    """Scatter plot one point per (method, city) at the mean across scenarios+
    episodes (and seeds, when multi-seed data is available)."""
    needed = ["method", "city", x_metric, y_metric]
    if not all(c in df.columns for c in needed):
        return False
    cols = needed + (["seed"] if "seed" in df.columns else [])
    sub = df[cols].dropna()
    if sub.empty:
        return False
    sub[x_metric] = pd.to_numeric(sub[x_metric], errors="coerce")
    sub[y_metric] = pd.to_numeric(sub[y_metric], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return False
    if "seed" in sub.columns and sub["seed"].nunique() > 1:
        per_seed = sub.groupby(["seed", "method", "city"]).agg(
            x=(x_metric, "mean"), y=(y_metric, "mean")).reset_index()
        agg = per_seed.groupby(["method", "city"]).agg(
            x=("x", "mean"), y=("y", "mean")).reset_index()
    else:
        agg = sub.groupby(["method", "city"]).agg(
            x=(x_metric, "mean"), y=(y_metric, "mean")).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    for m, g in agg.groupby("method"):
        ax.scatter(g["x"], g["y"], s=80, label=m, color=_color_for(m),
                   edgecolor="black", linewidth=0.5, alpha=0.85)
        for _, row in g.iterrows():
            ax.annotate(row["city"], (row["x"], row["y"]),
                        fontsize=7, alpha=0.65,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel(x_label + (" ↑" if x_dir == "higher" else (" ↓" if x_dir == "lower" else "")))
    ax.set_ylabel(y_label + (" ↑" if y_dir == "higher" else (" ↓" if y_dir == "lower" else "")))
    ax.set_title(f"Pareto: {x_label} vs {y_label}")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# ────────────────────────────────────────────────────────────────────────────
# Inter-seed analysis (only meaningful when ≥ 2 seeds in the data)
# (SEED_HEADLINE defined alongside the 5-axes group at the top of the file.)
# ────────────────────────────────────────────────────────────────────────────


def build_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method roll-up over seeds for the headline metrics.

    For each metric we report the mean of the per-seed means and the
    standard deviation across seeds (so the reader sees how reproducible
    the method is, not just where it ends up on average)."""
    if "seed" not in df.columns or df["seed"].nunique() < 2:
        return pd.DataFrame()
    metrics = [c for c, _, _ in SEED_HEADLINE if c in df.columns]
    if not metrics or "method" not in df.columns:
        return pd.DataFrame()
    sub = df[["seed", "method"] + metrics].copy()
    for c in metrics:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    per_seed = sub.groupby(["method", "seed"])[metrics].mean().reset_index()
    out = per_seed.groupby("method")[metrics].agg(["mean", "std"])
    out.columns = [f"{m}__{stat}" for m, stat in out.columns]
    return out.reset_index()


def seed_reproducibility_plot(df: pd.DataFrame, out_path: Path) -> bool:
    """One panel per headline metric. Inside each panel: one cluster per
    method showing every individual seed as a dot plus the per-method mean
    as a horizontal tick. This is the chart that answers "do my numbers
    move when I change the seed?" without any scenario-scale confound."""
    if "seed" not in df.columns or df["seed"].nunique() < 2:
        return False
    metrics = [m for m in SEED_HEADLINE if m[0] in df.columns]
    if not metrics:
        return False

    methods = sorted(df["method"].dropna().unique())
    seeds   = sorted(df["seed"].dropna().unique())
    n = len(metrics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 3.6 * nrows), squeeze=False)
    seed_marker = ["o", "s", "D", "^", "v", "P", "X"]

    for ax_idx, (col, label, direction) in enumerate(metrics):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        sub = df[["seed", "method", col]].dropna()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=[col])
        per_seed = sub.groupby(["method", "seed"])[col].mean().reset_index()
        method_mean = per_seed.groupby("method")[col].mean()

        for xi, m in enumerate(methods):
            mrows = per_seed[per_seed["method"] == m]
            for _, r in mrows.iterrows():
                mk = seed_marker[seeds.index(r["seed"]) % len(seed_marker)]
                ax.scatter(xi, r[col], marker=mk, s=55,
                            color=_color_for(m), edgecolor="black", linewidth=0.4,
                            label=f"seed {int(r['seed'])}" if xi == 0 else "_nolegend_")
            if m in method_mean.index:
                ax.hlines(method_mean[m], xi - 0.25, xi + 0.25,
                          colors="black", linewidth=2.0)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        arrow = " ↑" if direction == "higher" else (" ↓" if direction == "lower" else "")
        ax.set_title(f"{label}{arrow}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="best", ncol=min(3, len(seeds)))

    # blank any leftover subplot
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle("Inter-seed reproducibility (dots = per-seed mean, bar = method mean)",
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def seed_scenario_dispersion_plot(df: pd.DataFrame, metric: str, label: str,
                                   direction: str, out_path: Path) -> bool:
    """Heatmap method × scenario showing the inter-seed range (max−min of
    the per-seed means) for a given metric. Highlights where seed choice
    matters most. Lower = more reproducible."""
    if "seed" not in df.columns or df["seed"].nunique() < 2:
        return False
    if metric not in df.columns or "scenario" not in df.columns:
        return False
    sub = _clean_numeric(df, ["seed", "method", "scenario", metric], metric)
    if sub.empty:
        return False
    per_seed = sub.groupby(["seed", "method", "scenario"])[metric].mean().reset_index()
    rng = per_seed.groupby(["method", "scenario"])[metric].agg(
        lambda v: float(np.max(v) - np.min(v))).reset_index(name="range")
    piv = rng.pivot_table(index="scenario", columns="method", values="range")
    if piv.empty:
        return False

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(piv.columns) + 2),
                                     max(3.5, 0.55 * len(piv.index) + 1.5)))
    im = ax.imshow(piv.values, aspect="auto", cmap="magma_r")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    arrow = " (lower = more reproducible)"
    ax.set_title(f"Inter-seed range of {label}{arrow}")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if pd.isna(v):
                continue
            txt = f"{v:.2f}" if abs(v) < 1000 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color="white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def seed_cv_plot(df: pd.DataFrame, out_path: Path) -> bool:
    """Coefficient-of-variation of per-seed means, one bar per method,
    one cluster per headline metric. Reads as "lower = more stable across
    seeds"."""
    if "seed" not in df.columns or df["seed"].nunique() < 2:
        return False
    metrics = [m for m in SEED_HEADLINE if m[0] in df.columns]
    if not metrics:
        return False

    methods = sorted(df["method"].dropna().unique())
    cv_rows = []
    for col, label, _ in metrics:
        sub = df[["seed", "method", col]].dropna()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=[col])
        per_seed = sub.groupby(["method", "seed"])[col].mean().reset_index()
        stats = per_seed.groupby("method")[col].agg(["mean", "std"])
        for m in methods:
            if m in stats.index and stats.loc[m, "mean"] not in (0, None) and not pd.isna(stats.loc[m, "mean"]):
                cv = abs(stats.loc[m, "std"] / stats.loc[m, "mean"])
            else:
                cv = np.nan
            cv_rows.append({"metric": label, "method": m, "cv": cv})
    cv_df = pd.DataFrame(cv_rows)

    labels = [m[1] for m in metrics]
    x = np.arange(len(labels))
    bw = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(labels) + 4), 4.5))
    for i, m in enumerate(methods):
        vals = cv_df[cv_df["method"] == m].set_index("metric")["cv"].reindex(labels).values
        ax.bar(x + i * bw, vals, bw, label=m, color=_color_for(m),
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x + bw * (len(methods) - 1) / 2)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Coefficient of variation across seeds (lower = more stable)")
    ax.set_title("Inter-seed variability per method")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# ────────────────────────────────────────────────────────────────────────────
# Summary table
# ────────────────────────────────────────────────────────────────────────────
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, city) with the headline metrics from each axis."""
    headline = [
        # capacity
        "throughput_rate", "accept_rate",
        # quality
        "latency_mean", "delivery_route_efficiency",
        "agent_completed_gini",
        # environment
        "mean_congestion", "mean_bpr_along_route",
        "total_fleet_distance_m",
        # operational
        "wallclock_ms", "compute_time_per_task_ms",
    ]
    cols = [c for c in headline if c in df.columns]
    if not cols or "method" not in df.columns or "city" not in df.columns:
        return pd.DataFrame()
    key = ["method", "city"]
    has_seed = "seed" in df.columns and df["seed"].nunique() > 1
    src_cols = (["seed"] if has_seed else []) + key + cols
    sub = df[src_cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    if has_seed:
        # collapse episodes → per-seed mean, then average across seeds, so
        # every seed contributes equally even if their episode counts differ.
        per_seed = sub.groupby(["seed"] + key)[cols].mean().reset_index()
        agg = per_seed.groupby(key)[cols].mean().reset_index()
    else:
        agg = sub.groupby(key)[cols].mean().reset_index()
    return agg.sort_values(["city", "method"]).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# Headline comparison: throughput / latency / BPR side-by-side
# ────────────────────────────────────────────────────────────────────────────
# The compact "method ranking" table — the three axes that matter most for the
# LGPDP comparison shown TOGETHER: capacity (throughput ↑), delivery quality
# (latency ↓) and environmental cost (mean BPR along route ↓). Plus accept_rate
# as a sanity column. Aggregation is seed-aware (episodes → per-seed mean →
# cross-seed mean) so every seed weighs equally regardless of its episode count.
HEADLINE_COMPARE = [
    ("throughput_rate",       "throughput", "higher"),
    ("latency_mean",          "latency",    "lower"),
    ("mean_bpr_along_route",  "bpr",        "lower"),
    ("accept_rate",           "accept",     "higher"),
]


def _seed_aware_mean(df: pd.DataFrame, metric: str, by: list[str]) -> pd.DataFrame:
    """Mean of `metric` grouped by `by`, collapsing episodes → per-seed mean →
    cross-seed mean when ≥2 seeds are present (equal weight per seed). With a
    single seed this is just the grouped mean across episodes."""
    keep = (["seed"] if "seed" in df.columns else []) + by + [metric]
    keep = [c for c in keep if c in df.columns]
    if metric not in keep or not all(b in df.columns for b in by):
        return pd.DataFrame(columns=by + [metric])
    sub = df[keep].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=[metric])
    if sub.empty:
        return pd.DataFrame(columns=by + [metric])
    if "seed" in sub.columns and sub["seed"].nunique() > 1:
        per_seed = sub.groupby(["seed"] + by)[metric].mean().reset_index()
        return per_seed.groupby(by)[metric].mean().reset_index()
    return sub.groupby(by)[metric].mean().reset_index()


def build_headline_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method table: throughput | latency | bpr | accept (seed-aware),
    ranked by throughput descending. The headline ranking table."""
    if "method" not in df.columns:
        return pd.DataFrame()
    out = None
    for col, name, _ in HEADLINE_COMPARE:
        if col not in df.columns:
            continue
        m = _seed_aware_mean(df, col, ["method"]).rename(columns={col: name})
        out = m if out is None else out.merge(m, on="method", how="outer")
    if out is None or out.empty:
        return pd.DataFrame()
    if "throughput" in out.columns:
        out = out.sort_values("throughput", ascending=False)
    return out.reset_index(drop=True)


def build_scenario_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """method × scenario matrix of the seed-aware mean of `metric`."""
    if "scenario" not in df.columns:
        return pd.DataFrame()
    agg = _seed_aware_mean(df, metric, ["method", "scenario"])
    if agg.empty:
        return pd.DataFrame()
    return agg.pivot(index="method", columns="scenario", values=metric)


def emit_headline_tables(df: pd.DataFrame, out_dir: Path, title: str) -> None:
    """Write + print the headline thr/lat/bpr comparison and the per-scenario
    matrices (throughput, latency, BPR). `out_dir` is created if missing."""
    headline = build_headline_comparison(df)
    if headline.empty:
        print(f"  [skip] {title}: no usable rows")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    headline.round(3).to_csv(out_dir / "headline_throughput_latency_bpr.csv",
                             index=False)
    print(f"\n  === {title}: headline (ranked by throughput) ===")
    print(headline.round(3).to_string(index=False))

    for metric, short in (("throughput_rate",      "throughput"),
                          ("latency_mean",         "latency"),
                          ("mean_bpr_along_route", "bpr")):
        mat = build_scenario_matrix(df, metric)
        if mat.empty:
            continue
        # Order rows by the overall throughput ranking for readability.
        order = [m for m in headline["method"] if m in mat.index]
        mat = mat.reindex(order)
        mat.round(3).to_csv(out_dir / f"matrix_{short}__method_x_scenario.csv")
    print(f"  [csv]  → {out_dir}")


def load_results_dir(results_dir: Path) -> pd.DataFrame:
    """Load + unify all per-seed training-schema CSVs from a results directory
    (e.g. Option_O_results/episodes_seedNN.csv). Reads the per-seed sources
    DIRECTLY so the comparison always reflects the actual final run in that
    folder — robust to a stale `episodes_all_seeds_unified.csv`."""
    import re
    frames = []
    for f in sorted(Path(results_dir).glob("episodes_seed*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {f.name}: {type(e).__name__}: {e}")
            continue
        if d.empty:
            continue
        d = _unify_training(d)
        if "seed" not in d.columns:
            m = re.search(r"seed(\d+)", f.stem)
            d["seed"] = int(m.group(1)) if m else -1
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Cross-method analysis ===")
    print(f"Output → {OUT_DIR}")

    df = load_all()
    if "method" not in df.columns:
        print("[FATAL] unified frame missing 'method' column.")
        sys.exit(1)
    if "city" not in df.columns:
        print("[FATAL] unified frame missing 'city' column.")
        sys.exit(1)
    if "scenario" not in df.columns:
        df["scenario"] = "(unknown)"

    print(f"\nLoaded {len(df)} rows total")
    print(f"  Methods : {sorted(df['method'].dropna().unique())}")
    print(f"  Cities  : {sorted(df['city'].dropna().unique())}")
    print(f"  Sources : {sorted(df['source'].dropna().unique())}")

    # ── Axis-wise plots (5 axes: Throughput / Quality / Impact / Wallclock / Seeds)
    for axis_name, (subdir, metrics) in ALL_AXES.items():
        print(f"\n--- Axis: {axis_name} ---")
        axis_dir = OUT_DIR / subdir
        for col, label, direction in metrics:
            slug = col
            bar_ok = bar_plot_metric(df, col, label, direction,
                                      axis_dir / f"{slug}__by_city.png",
                                      group_by="city")
            heat_ok = heatmap_metric(df, col, label, direction,
                                      axis_dir / f"{slug}__heatmap.png")
            # Per-scenario stratification (helpful for stress vs over_fleet)
            scen_ok = False
            if "scenario" in df.columns:
                scen_ok = bar_plot_metric(df, col, label + " (per scenario)",
                                           direction,
                                           axis_dir / f"{slug}__by_scenario.png",
                                           group_by="scenario")
            tags = "".join(["B" if bar_ok else "-",
                            "H" if heat_ok else "-",
                            "S" if scen_ok else "-"])
            print(f"  [{tags}] {col:<32} → {label}")

    # ── Pareto / trade-off plots ─────────────────────────────────────────
    print("\n--- Pareto trade-offs ---")
    pareto_dir = OUT_DIR / "pareto"
    pareto_specs = [
        # (x_metric, x_label, x_dir,
        #  y_metric, y_label, y_dir, filename)
        ("throughput_rate", "Throughput rate", "higher",
         "latency_mean",    "Mean latency",    "lower",
         "throughput_vs_latency.png"),
        ("throughput_rate", "Throughput rate",     "higher",
         "mean_congestion", "Mean congestion",     "lower",
         "throughput_vs_congestion.png"),
        ("latency_mean",    "Mean latency",        "lower",
         "mean_congestion", "Mean congestion",     "lower",
         "latency_vs_congestion.png"),
        ("throughput_rate", "Throughput rate",         "higher",
         "agent_completed_gini", "Fairness (Gini)",   "lower",
         "throughput_vs_fairness.png"),
        ("throughput_rate",          "Throughput rate", "higher",
         "total_fleet_distance_m",   "Total fleet distance (m)", "lower",
         "throughput_vs_fleet_distance.png"),
        ("throughput_rate",          "Throughput rate",          "higher",
         "compute_time_per_task_ms", "Compute time per task (ms)", "lower",
         "throughput_vs_compute_cost.png"),
    ]
    for x, xl, xd, y, yl, yd, fn in pareto_specs:
        ok = pareto_plot(df, x, xl, xd, y, yl, yd, pareto_dir / fn)
        tag = "OK" if ok else "skip"
        print(f"  [{tag}] {x} vs {y}")

    # ── Summary table ────────────────────────────────────────────────────
    summary = build_summary(df)
    if not summary.empty:
        out_csv = OUT_DIR / "summary" / "summary_method_x_city.csv"
        summary.to_csv(out_csv, index=False)
        print(f"\nSummary table → {out_csv}")
        # also a per-method roll-up (mean across cities) — handy text snapshot
        per_method = summary.drop(columns=["city"]).groupby("method").mean(
            numeric_only=True).round(4)
        per_method.to_csv(OUT_DIR / "summary" / "summary_per_method.csv")
        print(per_method.to_string())

    # ── Headline thr / lat / BPR comparison (the compact ranking table) ──
    # Computed on the loaded (unified) frame, AND — robustly — on a direct read
    # of Option_O_results so the final RL/TP/RMCA run always shows up even if
    # episodes_all_seeds_unified.csv is stale (missing RMCA / older numbers).
    print("\n--- Headline comparison (throughput / latency / BPR) ---")
    emit_headline_tables(df, OUT_DIR / "summary", "Loaded data (unified source)")

    oo = load_results_dir(RESULTS / "Option_O_results")
    if not oo.empty:
        if "scenario" not in oo.columns:
            oo["scenario"] = "(unknown)"
        n_oo_seeds = oo["seed"].nunique() if "seed" in oo.columns else 1
        print(f"\n  [Option O direct read: {len(oo)} rows, {n_oo_seeds} seed(s), "
              f"methods={sorted(oo['method'].dropna().unique())}]")
        emit_headline_tables(oo, OUT_DIR / "summary" / "option_o_final",
                             "Option O FINAL (RL / TP / RMCA, direct read)")

    # ── Inter-seed analysis (only when ≥ 2 seeds in the data) ────────────
    n_seeds = df["seed"].nunique() if "seed" in df.columns else 0
    if n_seeds >= 2:
        print(f"\n--- Inter-seed analysis ({n_seeds} seeds: {sorted(df['seed'].dropna().unique())}) ---")
        seeds_dir = OUT_DIR / "05_seeds"
        seed_summary = build_seed_summary(df)
        if not seed_summary.empty:
            csv_path = seeds_dir / "summary_per_method_with_seed_std.csv"
            seed_summary.round(4).to_csv(csv_path, index=False)
            print(f"  [csv]  → {csv_path}")
        repro_ok = seed_reproducibility_plot(df, seeds_dir / "reproducibility_dots.png")
        print(f"  [{'OK' if repro_ok else 'skip'}] reproducibility_dots.png")
        cv_ok = seed_cv_plot(df, seeds_dir / "seed_cv_per_method.png")
        print(f"  [{'OK' if cv_ok else 'skip'}] seed_cv_per_method.png")
        # Per-(method × scenario) dispersion heatmaps — one per headline
        # metric. Same headline set as the dots/CV plots.
        for col, label, direction in SEED_HEADLINE:
            disp_ok = seed_scenario_dispersion_plot(
                df, col, label, direction,
                seeds_dir / f"dispersion__{col}__by_scenario.png")
            print(f"  [{'OK' if disp_ok else 'skip'}] dispersion__{col}__by_scenario.png")
    else:
        print(f"\n--- Inter-seed analysis: skipped (only {n_seeds} seed(s) in data) ---")

    print(f"\nDone. Output tree:")
    for sub in sorted(OUT_DIR.iterdir()):
        if sub.is_dir():
            n = sum(1 for _ in sub.rglob("*.png")) + sum(1 for _ in sub.rglob("*.csv"))
            print(f"  {sub.name:<14} : {n} files")


if __name__ == "__main__":
    main()
