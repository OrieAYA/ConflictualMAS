# Accepted / Reference Results

Working MAPPO results from before the current code regression.

## Files

| File | Date | Description |
|---|---|---|
| `policy_seed42_mappo_working_2026-05-17.bin` | 2026-05-17 00:40 | **Best working MAPPO checkpoint** — selective policy, accept 36-63% by city |
| `policy_seed43_mappo_working_2026-05-16.bin` | 2026-05-16 | Earlier seed-43 MAPPO checkpoint, also working |
| `summary_mappo_working_2026-05-17.csv` | 2026-05-17 | Per-(city, phase, policy) summary with the working MAPPO numbers |
| `episodes_seed43_2026-05-16.csv` | 2026-05-16 | Per-episode log for seed 43 |
| `TrainingResults_2026-05-17.txt` | 2026-05-17 | Console output of the training run |

## Performance baseline (the run from 2026-05-17)

| City | Phase | Accept rate | Throughput | vs Greedy |
|---|---|---|---|---|
| Tokyo_Small | train MAPPO | 36.1% | 0.201 | — |
| Tokyo_Small | eval MAPPO | 38.4% | 0.217 | Greedy 0.128 (+70%) |
| Tokyo_Medium | train MAPPO | 47.9% | 0.157 | — |
| Tokyo_Medium | eval MAPPO | 52.2% | 0.191 | Greedy 0.075 (+155%) |
| Tokyo_Large | train MAPPO | 63.1% | 0.082 | — |
| Tokyo_Large | eval MAPPO | 64.1% | 0.093 | Greedy 0.031 (+200%) |

These results show MAPPO beating Greedy / Random / InsertionGreedy by 70-200% on throughput,
while learning a city-size-dependent acceptance policy (more selective in smaller cities,
more accepting in larger ones).

## Compatibility note

The 2026-05-17 checkpoints were saved with `kPolicySz = 12` (12 features).
The current code uses `kPolicySz = 13` (added `deliverability` feature for lifelong support).

**These checkpoints are NOT directly loadable into the current code** without one of:
1. Rolling back `kPolicySz` to 12 in `ObjectiveDMPolicy.hpp` (loses the deliverability feature)
2. Writing a checkpoint migration: pad weights with zeros on the new input dimension
3. Re-training MAPPO with the current 13-feature architecture (and fixing whatever regressed)

The checkpoints are kept here as the **reference for what MAPPO can achieve** on this
problem when configured correctly. Use them to:
- Compare current training trajectory to the working trajectory
- Validate that the architecture (TAM + planning + RL) is sound — these numbers prove it
- Diagnose what changed between 2026-05-17 and the current code that broke MAPPO convergence

## Suspected regression sources (changes since 2026-05-17)

Investigated diff summary: 1153 insertions / 189 deletions across 12 files in Training/ and Policy/.

Top suspects:
1. **Scenario sampler change** — added stress_heavy regime (density 1.8-2.8) at 20% probability,
   shifting the gradient toward refuse-everything in mixed training
2. **`deliverability` feature** added — new input dim, but Xavier init may bias the policy
3. **Cross-policy interference** — IPPO and MAPPER training runs share the same EpisodeRunner
   and CongestionMap; running them in sequence may pollute the path cache / dynamic costs
4. **Buffer clearing semantics** — eval `clear_buffer()` calls may have edge cases that
   weren't there before
