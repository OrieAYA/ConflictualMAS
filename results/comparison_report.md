# Comparison Report - phase = `eval`

_Generated 2026-05-16 13:01 from 1 seed CSV(s)._
_Mapping to objective functions: research PDF slides 17, 44._

## Obj. 1 - Throughput (tasks_completed / tasks_appeared)

Higher is better. Mean +- std across eval episodes.

| Method | Tokyo_Large | Tokyo_Medium | Tokyo_Small |
|---|---|---|---|
| MAPPO | 0.1021 +- 0.0299 | 0.1414 +- 0.0215 | 0.2025 +- 0.0283 |
| TamAlwaysAccept | 0.0783 +- 0.0314 | 0.1462 +- 0.0295 | 0.2118 +- 0.0329 |
| InsertionGreedy | 0.0667 +- 0.0251 | 0.0873 +- 0.0175 | 0.1256 +- 0.018 |
| Greedy | 0.0192 +- 0.0098 | 0.0794 +- 0.0145 | 0.1132 +- 0.0222 |
| Random | 0.0314 +- 0.0092 | 0.0747 +- 0.0145 | 0.1142 +- 0.02 |

## Obj. 2 - Completeness latency (steps, pickup -> delivery)

Lower is better. `latency_mean` averaged across delivered tasks.

| Method | Tokyo_Large | Tokyo_Medium | Tokyo_Small |
|---|---|---|---|
| MAPPO | 1690.6 +- 196.6 | 1687.5 +- 194.7 | 1184.8 +- 113.6 |
| TamAlwaysAccept | 1913.5 +- 419.7 | 1582.6 +- 109.7 | 1240.7 +- 194.1 |
| InsertionGreedy | 1780.3 +- 237.6 | 1802.1 +- 184.6 | 1423.3 +- 136.4 |
| Greedy | 1997.8 +- 392.6 | 1894.9 +- 172.4 | 1707.1 +- 193.4 |
| Random | 2162.9 +- 271.2 | 2137.8 +- 127 | 1557.1 +- 163.6 |

### Obj. 2b - Latency per available agent

`latency_mean / mean_active_agents` (slide 44, `Completeness latency mean related to the number of available agents`).

| Method | Tokyo_Large | Tokyo_Medium | Tokyo_Small |
|---|---|---|---|
| MAPPO | 212.61 +- 31.1 | 225.87 +- 28.33 | 245.31 +- 14.2 |
| TamAlwaysAccept | 220.36 +- 43.65 | 238.97 +- 28.84 | 256.37 +- 30.22 |
| InsertionGreedy | 257.64 +- 37.11 | 242.51 +- 32.82 | 284.65 +- 27.29 |
| Greedy | 249.56 +- 42.68 | 254.72 +- 37.95 | 341.42 +- 38.67 |
| Random | 237.34 +- 35.62 | 273.84 +- 26.23 | 311.42 +- 32.71 |

## Obj. 3 - Task allocation: acceptance rate

Higher is not always better. Under capacity constraint, over-acceptance leads to unfinished tasks.
MAPPO is expected to find a task-aware sweet spot rather than max or min.

| Method | Tokyo_Large | Tokyo_Medium | Tokyo_Small |
|---|---|---|---|
| MAPPO | 0.6314 +- 0.0603 | 0.453 +- 0.0164 | 0.3669 +- 0.0409 |
| TamAlwaysAccept | 0.6526 +- 0.0462 | 0.4831 +- 0.0675 | 0.3675 +- 0.0426 |
| InsertionGreedy | 0.6114 +- 0.046 | 0.4017 +- 0.0518 | 0.2803 +- 0.027 |
| Greedy | 0.601 +- 0.0532 | 0.403 +- 0.0466 | 0.2755 +- 0.0363 |
| Random | 0.5628 +- 0.0419 | 0.3912 +- 0.0336 | 0.2743 +- 0.0339 |

## Obj. 4 - Computation cost ratio (wallclock vs `Greedy`)

Lower is better. Captures the `computation cost ratio` requirement on slide 17.

| Method | Tokyo_Large | Tokyo_Medium | Tokyo_Small |
|---|---|---|---|
| MAPPO | 9.72x | 6.42x | 7.32x |
| TamAlwaysAccept | 11.52x | 4.56x | 6.26x |
| InsertionGreedy | 1.41x | 1.71x | 1.49x |
| Greedy | 1x | 1x | 1x |
| Random | 0.52x | 0.91x | 0.8x |

## Bonus - Scalability (throughput across city sizes)

Tokyo_Small -> Tokyo_Medium (-> Tokyo_Large if available).
Shows degradation curve as graph size grows.

| Method | Tokyo_Large | Tokyo_Medium | Tokyo_Small |
|---|---|---|---|
| MAPPO | 0.1021 +- 0.0299 | 0.1414 +- 0.0215 | 0.2025 +- 0.0283 |
| TamAlwaysAccept | 0.0783 +- 0.0314 | 0.1462 +- 0.0295 | 0.2118 +- 0.0329 |
| InsertionGreedy | 0.0667 +- 0.0251 | 0.0873 +- 0.0175 | 0.1256 +- 0.018 |
| Greedy | 0.0192 +- 0.0098 | 0.0794 +- 0.0145 | 0.1132 +- 0.0222 |
| Random | 0.0314 +- 0.0092 | 0.0747 +- 0.0145 | 0.1142 +- 0.02 |

## Auxiliary - Agent utilisation

Fraction of steps with active agents. Helps disambiguate low-throughput cases (slack vs overload).

| Method | Tokyo_Large | Tokyo_Medium | Tokyo_Small |
|---|---|---|---|
| MAPPO | 0.5734 +- 0.0356 | 0.6611 +- 0.0294 | 0.6741 +- 0.0257 |
| TamAlwaysAccept | 0.6001 +- 0.0424 | 0.6018 +- 0.0309 | 0.6723 +- 0.0396 |
| InsertionGreedy | 0.5132 +- 0.0681 | 0.6587 +- 0.0429 | 0.6959 +- 0.0347 |
| Greedy | 0.5604 +- 0.0452 | 0.6697 +- 0.0158 | 0.7159 +- 0.0062 |
| Random | 0.6303 +- 0.0493 | 0.6881 +- 0.0223 | 0.7244 +- 0.0101 |

## Methodology - episode counts

| Method | Tokyo_Large | Tokyo_Medium | Tokyo_Small |
|---|---|---|---|
| MAPPO | 6 | 6 | 6 |
| TamAlwaysAccept | 6 | 6 | 6 |
| InsertionGreedy | 6 | 6 | 6 |
| Greedy | 6 | 6 | 6 |
| Random | 6 | 6 | 6 |

