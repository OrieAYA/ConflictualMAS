# `_analyze_cross_method.py` — usage

## What it does

Reads up to four CSV sources produced by Option O / Option G / earlier
runs and produces three sets of plots + a summary table along the
publication's three evaluation axes:

| Axis | Question | Metrics |
|---|---|---|
| **CAPACITE** | How much can the method handle? | `throughput_rate`, `accept_rate`, `tasks_completed`, `value_throughput_rate` |
| **QUALITE** | How well are deliveries served? | `latency_mean`, `mean_wait_steps`, `mean_trip_steps`, `delivery_route_efficiency`, `mean_extra_steps_per_task`, `agent_completed_gini`, `agent_completed_std` |
| **ENVIRONNEMENT** | Network footprint? | `mean_congestion`, `congestion_variance`, `peak_load`, `mean_bpr_along_route`, `n_traversals_in_jam`, `route_congestion_exposure`, `time_lost_to_congestion`, `total_fleet_distance_m` |

Plus six Pareto/trade-off scatter plots: throughput vs latency, vs
congestion, vs fairness, vs fleet distance, vs compute cost ; latency vs
congestion.

## Required CSV inputs (auto-skipped if missing)

| Name | Path | Schema |
|---|---|---|
| `RL_train`  | `results/Evaluation Files/episodes_seed42_RL.csv`                           | TrainingLogger |
| `TP_train`  | `results/Evaluation Files/episodes_seed42_TP.csv`                           | TrainingLogger |
| `ORL_gen`   | `results/Option_O_results/episodes_seed42.csv`                              | TrainingLogger |
| `HAPC_gen`  | `results/Option_G_results/sota_standalone_O/sota_standalone_seed42.csv`     | SolverCSVLogger |

The script auto-handles both schemas. Re-running incrementally as new CSVs
appear is safe — it just refreshes the graphs.

## Install Python + run

This box has no real Python yet (only the MS Store launcher). To install
and run:

```powershell
# 1) install Python 3.11+ from python.org (default path is fine)
# 2) install the 3 deps
py -m pip install pandas matplotlib numpy

# 3) run the analysis
cd C:\ConflictualMAS
py results\_analyze_cross_method.py
```

Outputs land under `results/Analysis_Cross_Method/`:

```
Analysis_Cross_Method/
├── capacity/        # throughput, acceptance, completion-count bars/heatmaps
├── quality/         # latency, wait, route-efficiency, Gini, etc.
├── environment/     # congestion, BPR, jam traversals, fleet distance
├── pareto/          # throughput-vs-latency etc. trade-off scatter plots
└── summary/
    ├── summary_method_x_city.csv
    └── summary_per_method.csv
```

## Run in parallel with O / G

The script ONLY READS the CSVs; it never touches them. Safe to launch in
its own terminal while Option O and Option G are still writing — re-run
whenever you want a fresh snapshot.

```powershell
# Terminal 1 :  Option O   (writes to Option_O_results/)
# Terminal 2 :  Option G   (writes to Option_G_results/)
# Terminal 3 :  this script (reads only)
py results\_analyze_cross_method.py
```
