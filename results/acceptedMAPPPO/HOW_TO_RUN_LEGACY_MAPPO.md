# How to run the legacy 12-feature MAPPO

The branch `mappo-legacy-12d-may14` (commit `86d33c6`) contains the exact code
that produced the working MAPPO results of 2026-05-17.

## Architecture of the legacy branch

- `kPolicySz = 12` (the `deliverability` feature does not exist)
- No scenario sampler — all training episodes use the city's nominal phase config
- No IPPO, no MAPPER — only MAPPO is trained
- 5 eval modes: MAPPO, TamAlwaysAccept, Greedy, Random, InsertionGreedy
- No stress eval, no generalize eval
- `max_tasks_per_agent`: 5/6/8 (Small/Medium/Large)
- Phase task density: 60/100/140 (Small), 100/150/200 (Medium), 120/200/280 (Large)
- Reward shaping: `pickup_reward_frac=0.3`, `refuse_penalty_w=0.15`, `unfinished_factor=1.0`

## To checkout and run

**Important: do NOT do this while the current training is running** — switching
branches changes the source files and breaks any rebuild.

```bash
# After the current training has finished (checkpoints saved):
git stash push -m "current-work-snapshot"   # safekeep uncommitted changes
git checkout mappo-legacy-12d-may14         # switch to the legacy code
# Build and run as before; the policy_seed42_mappo_working_2026-05-17.bin
# from results/accepted/ is loadable directly:
#   cfg.load_policy = true;
#   cfg.policy_path = "results/accepted/policy_seed42_mappo_working_2026-05-17.bin";

# When done, return to the modern code:
git checkout main
git stash pop                                # restore uncommitted changes
```

## What you reproduce

Running the legacy branch with the saved checkpoint reproduces these
test-set numbers exactly (these are the validated reference numbers,
see `summary_mappo_working_2026-05-17.csv`):

| City | Phase | MAPPO accept | MAPPO throughput | Greedy throughput | Speedup |
|---|---|---|---|---|---|
| Tokyo_Small | eval | 38.4% | 0.217 | 0.128 | +70% |
| Tokyo_Medium | eval | 52.2% | 0.191 | 0.075 | +155% |
| Tokyo_Large | eval | 64.1% | 0.093 | 0.031 | +200% |

## Compatibility with the current branch

- The 12-feature checkpoints are NOT loadable into the 13-feature current code.
- The legacy branch can be rebuilt independently as long as no source from
  the current branch is mixed in.
- For paper comparisons, treat MAPPO (legacy, 12-d) and MAPPER (current, 13-d)
  as two separate evaluations with documented architectural differences.
