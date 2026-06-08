#!/usr/bin/env python3
"""
Unify per-seed evaluation CSVs into the canonical `episodes_seed{N}_unified.csv`
schema (35 columns) used for cross-method publication analysis.

Inputs (any subset):
  - episodes_seed{N}.csv         : RL + TP rows (MultiCityTrainer)
                                    OR
    episodes_seed{N}_RL.csv      : RL-only (legacy split)
    episodes_seed{N}_TP.csv      : TP-only (legacy split)
  - sota_standalone_seed{N}.csv  : HAPC (or any solver registered by SolverCSVLogger)
                                    OR
    episodes_seed{N}_HAPC.csv    : legacy HAPC name

Output:
  - episodes_seed{N}_unified.csv (35 columns, header listed below)

Schema (unified):
  source,method,city,scenario,episode,n_agents,total_steps,
  tasks_appeared,tasks_completed,throughput_rate,accept_rate,
  latency_mean,latency_per_agent,agent_utilisation,
  mean_congestion,mean_trip_steps,mean_wait_steps,mean_road_pd_m,
  delivery_route_efficiency,mean_extra_steps_per_task,peak_load,
  congestion_variance,mean_bpr_along_route,n_traversals_in_jam,
  route_congestion_exposure,total_fleet_distance_m,
  agent_completed_gini,agent_completed_std,n_ghost_active_mean,
  mean_congestion_at_decision,wallclock_ms,
  compute_time_per_task_ms,compute_time_per_decision_us,
  capacity_violations,pairing_violations

source ∈ {"RL","TP","HAPC"}.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

UNIFIED_HEADER = [
    "source", "method", "city", "scenario", "episode",
    "n_agents", "total_steps", "tasks_appeared", "tasks_completed",
    "throughput_rate", "accept_rate",
    "latency_mean", "latency_per_agent", "agent_utilisation",
    "mean_congestion", "mean_trip_steps", "mean_wait_steps", "mean_road_pd_m",
    "delivery_route_efficiency", "mean_extra_steps_per_task", "peak_load",
    "congestion_variance", "mean_bpr_along_route", "n_traversals_in_jam",
    "route_congestion_exposure", "total_fleet_distance_m",
    "agent_completed_gini", "agent_completed_std", "n_ghost_active_mean",
    "mean_congestion_at_decision", "wallclock_ms",
    "compute_time_per_task_ms", "compute_time_per_decision_us",
    "capacity_violations", "pairing_violations",
]

# Phase column in the RL/TP CSV is e.g. "eval_normal_wave" or "generalize_stress_shock".
# Strip the leading "eval_" / "generalize_" so the scenario matches the SoTA CSV.
def _scenario_from_phase(phase: str) -> str:
    for pref in ("eval_", "generalize_", "train_", "warmup_"):
        if phase.startswith(pref):
            return phase[len(pref):]
    return phase

# Methods coming from the RL/TP combined CSV.
RL_METHODS = {"MAPPO", "FaithfulMAPPER", "MAPPER", "IPPO", "Hybrid"}
TP_METHODS = {"TokenPassing"}


def _convert_rl_tp(row: dict, n_eval_episodes: int = 2) -> dict:
    """Convert one episodes_seed{N}.csv row → unified row."""
    method = row["policy_mode"]
    if method in RL_METHODS:
        source = "RL"
    elif method in TP_METHODS:
        source = "TP"
    else:
        # Unknown: keep raw method name and tag source as "RL" by default.
        source = "RL"
    global_ep = int(row["global_episode"])
    return {
        "source":                       source,
        "method":                       method,
        "city":                         row["city"],
        "scenario":                     _scenario_from_phase(row["phase"]),
        "episode":                      global_ep % max(1, n_eval_episodes),
        "n_agents":                     row["n_agents_max"],
        "total_steps":                  row["total_steps"],
        "tasks_appeared":               row["tasks_appeared"],
        "tasks_completed":              row["tasks_completed"],
        "throughput_rate":              row["throughput_rate"],
        "accept_rate":                  row["accept_rate"],
        "latency_mean":                 row["latency_mean"],
        "latency_per_agent":            row["latency_per_agent"],
        "agent_utilisation":            row["agent_utilisation"],
        "mean_congestion":              row["mean_congestion"],
        "mean_trip_steps":              row["mean_trip_steps"],
        "mean_wait_steps":              row["mean_wait_steps"],
        "mean_road_pd_m":               row["mean_road_pd_m"],
        "delivery_route_efficiency":    row["delivery_route_efficiency"],
        "mean_extra_steps_per_task":    row["mean_extra_steps_per_task"],
        # peak_congestion column in RL/TP is the "peak load over time" metric →
        # renamed to peak_load in the unified schema (HAPC uses peak_load already).
        "peak_load":                    row["peak_congestion"],
        "congestion_variance":          row["congestion_variance"],
        "mean_bpr_along_route":         row["mean_bpr_along_route"],
        "n_traversals_in_jam":          row["n_traversals_in_jam"],
        "route_congestion_exposure":    row["route_congestion_exposure"],
        "total_fleet_distance_m":       row["total_fleet_distance_m"],
        "agent_completed_gini":         row["agent_completed_gini"],
        "agent_completed_std":          row["agent_completed_std"],
        "n_ghost_active_mean":          row["n_ghost_active_mean"],
        "mean_congestion_at_decision":  row["mean_congestion_at_decision"],
        "wallclock_ms":                 row["wallclock_ms"],
        "compute_time_per_task_ms":     row["compute_time_per_task_ms"],
        "compute_time_per_decision_us": row["compute_time_per_decision_us"],
        "capacity_violations":          row["capacity_violations"],
        "pairing_violations":           row["pairing_violations"],
    }


def _convert_sota(row: dict) -> dict:
    """Convert one sota_standalone_seed{N}.csv row → unified row.

    Tagged as source = "HAPC" by default (matches seed 42 convention for the
    HybridAdaptivePredictive solver). Any other solver names land here too with
    source "HAPC"; downstream analysis can re-tag by `method` if needed.
    """
    return {
        "source":                       "HAPC",
        "method":                       row["solver"],
        "city":                         row["city"],
        "scenario":                     row["scenario"],
        "episode":                      row["episode"],
        "n_agents":                     row["n_agents"],
        "total_steps":                  row["total_steps"],
        "tasks_appeared":               row["tasks_appeared"],
        "tasks_completed":              row["tasks_completed"],
        "throughput_rate":              row["throughput_rate"],
        "accept_rate":                  row["accept_rate"],
        "latency_mean":                 row["latency_mean"],
        "latency_per_agent":            row["latency_per_agent"],
        "agent_utilisation":            row["agent_utilisation"],
        "mean_congestion":              row["mean_congestion"],
        "mean_trip_steps":              row["mean_trip_steps"],
        "mean_wait_steps":              row["mean_wait_steps"],
        "mean_road_pd_m":               row["mean_road_pd_m"],
        "delivery_route_efficiency":    row["delivery_route_efficiency"],
        "mean_extra_steps_per_task":    row["mean_extra_steps_per_task"],
        "peak_load":                    row["peak_load"],
        "congestion_variance":          row["congestion_variance"],
        "mean_bpr_along_route":         row["mean_bpr_along_route"],
        "n_traversals_in_jam":          row["n_traversals_in_jam"],
        "route_congestion_exposure":    row["route_congestion_exposure"],
        "total_fleet_distance_m":       row["total_fleet_distance_m"],
        "agent_completed_gini":         row["agent_completed_gini"],
        "agent_completed_std":          row["agent_completed_std"],
        "n_ghost_active_mean":          row["n_ghost_active_mean"],
        "mean_congestion_at_decision":  row["mean_congestion_at_decision"],
        "wallclock_ms":                 row["wallclock_ms"],
        "compute_time_per_task_ms":     row["compute_time_per_task_ms"],
        "compute_time_per_decision_us": row["compute_time_per_decision_us"],
        "capacity_violations":          row["capacity_violations"],
        "pairing_violations":           row["pairing_violations"],
    }


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="seed number (e.g. 43)")
    ap.add_argument(
        "--in-dir",
        default=r"c:\ConflictualMAS\results\Evaluation Files",
        help="directory containing the source CSVs and where the unified file will be written",
    )
    ap.add_argument(
        "--n-eval-episodes",
        type=int,
        default=2,
        help="n_eval_episodes used by the training run "
             "(global_episode mod this value = inner episode index)",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    s = args.seed

    # Source files — try every known name; missing files are silently skipped.
    # Naming conventions encountered across seeds:
    #   seed 42: episodes_seed42_RL.csv + episodes_seed42_TP.csv (split)
    #            OR  episodes_seed42_RL_TP.csv (combined, legacy name)
    #   seed 43: episodes_seed43.csv (combined, current MultiCityTrainer name)
    #   seed 44: episodes_seed44_RL_TP.csv (combined, legacy name)
    # SoTA HAPC:
    #   seed 42: episodes_seed42_HAPC.csv (+ legacy _small variant)
    #   seed 43+ : sota_standalone_seed{N}.csv (current SolverCSVLogger name)
    rl_tp_combined  = in_dir / f"episodes_seed{s}.csv"
    rl_tp_legacy    = in_dir / f"episodes_seed{s}_RL_TP.csv"
    rl_split        = in_dir / f"episodes_seed{s}_RL.csv"
    tp_split        = in_dir / f"episodes_seed{s}_TP.csv"
    sota_standalone = in_dir / f"sota_standalone_seed{s}.csv"
    hapc_legacy     = in_dir / f"episodes_seed{s}_HAPC.csv"
    hapc_legacy_sm  = in_dir / f"episodes_seed{s}_HAPC_small.csv"

    unified_rows: list[dict] = []

    # 1) RL + TP from the combined or split files. Priority order:
    #    a) combined "episodes_seed{N}.csv"
    #    b) combined legacy "episodes_seed{N}_RL_TP.csv"
    #    c) split RL + TP files
    if rl_tp_combined.exists():
        for row in _read_csv(rl_tp_combined):
            unified_rows.append(
                _convert_rl_tp(row, n_eval_episodes=args.n_eval_episodes)
            )
    elif rl_tp_legacy.exists():
        for row in _read_csv(rl_tp_legacy):
            unified_rows.append(
                _convert_rl_tp(row, n_eval_episodes=args.n_eval_episodes)
            )
    else:
        for row in _read_csv(rl_split):
            unified_rows.append(
                _convert_rl_tp(row, n_eval_episodes=args.n_eval_episodes)
            )
        for row in _read_csv(tp_split):
            unified_rows.append(
                _convert_rl_tp(row, n_eval_episodes=args.n_eval_episodes)
            )

    # 2) SoTA standalone (HAPC) from the new or legacy filenames.
    for src in (sota_standalone, hapc_legacy, hapc_legacy_sm):
        for row in _read_csv(src):
            unified_rows.append(_convert_sota(row))

    if not unified_rows:
        print(f"[unify] no source rows found for seed {s} in {in_dir}", file=sys.stderr)
        return 2

    out_path = in_dir / f"episodes_seed{s}_unified.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_HEADER)
        writer.writeheader()
        for r in unified_rows:
            writer.writerow(r)

    # Brief summary on stdout — counts per (source, method) so we can sanity-check.
    counts: dict[tuple[str, str], int] = {}
    for r in unified_rows:
        key = (r["source"], r["method"])
        counts[key] = counts.get(key, 0) + 1
    print(f"[unify] wrote {len(unified_rows)} rows → {out_path}")
    for (src, m), n in sorted(counts.items()):
        print(f"  {src:<5} {m:<25} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())