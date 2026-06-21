# ConflictualMAS — Architecture & Workflows

Référence du fonctionnement **réel** du code (déduit de la logique, pas des
commentaires). Problème : Congestion-Aware Lifelong General Pickup & Delivery
(CA-L-GPDP). Article de référence : *An Agent-Based Architecture for the
CA-L-GPDP*. Les références sont données en `fichier:ligne`.

---

## 1. Structures de données de l'environnement

### 1.1 Graphe routier — `src/Environment/GeoBox/Box.hpp`

Le graphe est une **liste d'adjacence implicite** portée par 3 tables de
`MyData` (`Box.hpp:79`) :

| Champ | Type | Réf |
|-------|------|-----|
| `nodes` | `unordered_map<id, Point>` | `Box.hpp:153` |
| `ways` | `unordered_map<id, Way>` | `Box.hpp:154` |
| `objective_groups` | `unordered_map<int, ObjectiveGroup>` | `Box.hpp:155` |

**Parcours** : le voisin d'un node `cur` via une arête `w` est
`w.node1_id == cur ? w.node2_id : w.node1_id`. Implémenté dans
`ObjectiveGroupCache::discover_step` (`ObjectiveCache.cpp:170`) et
`PDPServerMemory::time_dependent_astar` (`ObjectiveCache.cpp:561`). C'est un
graphe non orienté, poids = `distance_meters` (statique) ou BPR (dynamique).

### 1.2 Congestion — `src/Environment/Congestion/CongestionMap.hpp`

`CongestionParams` (`CongestionMap.hpp:8`) : `horizon=1800` (`13`),
`bpr_alpha=0.15` (`14`), `bpr_beta=4` (`15`), `capacity_per_meter=0.05` (`16`),
`load_per_agent` (`23`).

État interne :
- `load_ : unordered_map<way_id, unordered_map<step_t, int>>`
  (`CongestionMap.hpp:118`) — **map temporelle creuse** : charge par
  (arête, pas de temps). Les entrées à 0 sont effacées (`update_load`,
  `CongestionMap.cpp:12`).
- `t_now_` (`119`) — horloge ; `advance` purge `t < t_now_`
  (`CongestionMap.cpp:97`).

Poids BPR : `adjusted_cost(way, base, dist, t, self_weight)`
(`CongestionMap.cpp:56`) =
`base · (1 + α·((load(way,t) − self_weight)/cap)^β)`, `cap = dist·0.05`
(min 1). `self_weight` permet à un agent d'exclure son propre poids (cf. §3.2).
`traversal_steps` (`CongestionMap.cpp:83`) = `ceil(adjusted_cost / speed)`,
autorité unique partagée par planning, commit et mouvement.

### 1.3 Trafic exogène — `src/Environment/Congestion/GhostTrafficController.hpp`

`CongestionProfile` (`:29`) : `Flat, RampUpDown, ShockBurst, BuildingUp, Wave`.
`Config` (`:70`) : `n_max` (`71`), `window_steps` (`73`),
`hot_way_fraction` (`74`), `density_per_hot_way` (`82`), `profile` (`90`).
État : `hot_ways_ : vector<way_id>` (`120`), `active_ : vector<GhostLoad>`
(`127`). Les ghosts écrivent dans `load_` via `add_ghost_load`
(`CongestionMap.cpp:37`), exactement comme les agents.

### 1.4 Path caches (meta-graphe inter-objectifs) — `ObjectiveCache.hpp`

`ObjectivePath` (`:14`) — un plus court chemin **statique** mémoïsé :
- `node_a / node_b` canoniques (min/max, `15-16`)
- `edges`, `nodes` (séquence ordonnée, `17-18`)
- `cost` (distance statique, `20`), `dynamic_cost` (temps BPR TD-A*, `21`),
  `dynamic_step` (pas du dernier calcul dynamique, `22`)

`ObjectiveGroupCache` (`:50`) :
- `paths_ : unordered_map<PathKey, ObjectivePath>` (`127`),
  `PathKey = pair<id,id>` (`52`)
- `objective_ids_ : unordered_set<id>` (`128`)
- `search_states_` (`141`) + `search_states_td_` (`144`) — **frontières
  Dijkstra incrémentales** (statique et temporelle), persistées entre appels
  pour reprendre l'expansion.

`PDPServerMemory` (`:158`) : `group_caches_ : unordered_map<int,
ObjectiveGroupCache>` (`262`). On traverse les plus courts chemins entre nodes
objectifs comme un graphe à part (pas les routes brutes) ; leur congestion se
lit dynamiquement sans modifier leur structure statique.

### 1.5 Fenêtres d'occupation — `src/DMASforPD/Utility/TimedPath.hpp`

`TimedPath` (`:20`) : `from_node/to_node`, `edge_ids` (`27`),
`edge_steps` (`28`, durée BPR par arête), `prefix_steps` (`29`, cumul →
`abs_entry`/`abs_exit`). Sert à projeter le plan d'un agent en fenêtres
`(arête, [t_entrée, t_sortie])` sur `load_`.

---

## 2. Le Manager — `src/DMASforPD/GlobalMemory/GlobalMemory.hpp`

`PDPGlobalMemory` détient l'environnement, **toutes les infos système**, et
fournit les **prédictions** de congestion (il ne subit pas le poids ; il
l'estime).

Membres publics :
- `geo_box&` (`:81`), `pathfinder&` (`:82`), `server_memory` (path caches, `83`),
  `congestion_map` (`84`), `task_agent` (`85`)
- listes de tâches `available/allocated/finished : vector<PDPTask*>` (`89-91`)
- `cur_global_state[20]` (`168`, état système publié par l'EpisodeRunner),
  `active_policy` (`163`), `region_grid` (`174`, heatmap densité×congestion)

Membres privés :
- `current_time_` (`307`)
- `tasks_ : unordered_map<int, PDPTask>` (`310`, stockage propriétaire ;
  adresses stables → les `PDPTask*` des listes restent valides)
- `node_to_task_id_ : unordered_map<node_id, int>` (`313`, lookup O(1)
  node→tâche)
- `delivery_agents_ : unordered_map<int, DeliveryAgent*>` (`316`)
- `committed_loads_ : unordered_map<int, vector<LoadWindow>>` (`328`),
  `LoadWindow{way, t_lo, t_hi, weight}` (`323`) — **ledger exact** des
  incréments réellement appliqués à `load_` par chaque agent, bornes clampées
  à la fenêtre `[t_now, t_now+horizon]` au moment du add, pour un remove
  parfaitement symétrique (`register_committed_plan` / `unregister_committed_plan`,
  `GlobalMemory.cpp`).

Prédictions de congestion fournies dans tout contexte temporel :
`adjusted_cost` / `bpr_path_cost` (`GlobalMemory.cpp`) /
`time_dependent_astar` (TD-A*, `ObjectiveCache.cpp:561`) — coût = temps BPR
estimé à l'instant d'arrivée prédit sur chaque arête.

**Reset** (`reset_episode`, `GlobalMemory.cpp`) : vide `tasks_`, listes,
`node_to_task_id_`, `committed_loads_`, appelle `congestion_map.reset()`, remet
`current_time_=0`, `region_grid.reset_episode()`. La **géométrie statique** de
`paths_` est conservée (donnée routière déterministe, indépendante de la
méthode) ; seuls `dynamic_cost`/`dynamic_step` sont invalidés
(`ObjectiveGroupCache::episode_reset`). Rejouer un même scénario sur une autre
méthode ne superpose donc aucun effet.

---

## 3. Agents

### 3.1 Task agent + TAM — `src/DMASforPD/TaskAgent/`

Un `TaskAgent` instancie un `TaskAllocationModule` (TAM) par tâche
(`on_new_task`, `TaskAgent.cpp`) et le détruit à la résolution (`erase_tam`).

`TaskAllocationModule` (`TaskAllocationModule.hpp:52`) :
- `matrix_ : unordered_map<int, AgentEntry>` ; `AgentEntry` (`:101`) :
  `pickup_cost`, `delivery_cost`, `pickup_node`, `delivery_node`,
  `idle_from_pickup`.
- `candidates_`, `bids_`, `scores_` alignés ; `max_search_cost_` (budget).

Logique (`step`, `TaskAllocationModule.cpp:147`) :
1. Deux Dijkstra incrémentaux depuis pickup et delivery (distance statique).
   Les **positions idle ne sont passées qu'à l'expansion pickup**
   (`TaskAllocationModule.cpp` — `discover_step(pickup, …, idle_positions_)` vs
   `discover_step(delivery, …, {})`).
2. Candidat valide (`collect_candidates`, `:47`) :
   - **idle** découvert côté pickup (`idle_from_pickup`), coût = `pickup_cost` ;
     **ou**
   - **busy** avec un node objectif côté pickup ET côté delivery, et
     `orientation_ok` = node-pickup **avant** node-delivery dans son plan
     (`TaskAllocationModule.cpp`, fonction `orientation_ok`).
3. Au 1er candidat (coût `x`), budget = `x · ratio(x)` (`ratio`, `:14`).
4. `finalise` (`:69`) : chaque candidat `bid_for_task` → (μ, accept). Gagnant =
   **argmax μ parmi les bidders** ; égalité de μ → coût d'insertion le plus
   faible (candidats triés croissant). Personne ne bid → `force_assign`
   (argmax μ global, marqué `forced_`) ou `deferred_`.

### 3.2 Delivery agent — `src/DMASforPD/DeliveryAgent/DeliveryAgent.hpp`

`DeliveryAgent` (`:90`) :
- `agent_id` (`92`), `status : AgentStatus{Idle,Active,Done}` (`93` / enum `:15`),
  `current_node` (`94`), `max_capacity` (`105`)
- `in_transit : optional<EdgeTransit>` (`95` / `EdgeTransit :18`),
  `edge_cursor : optional<EdgeCursor>` (`96` / `EdgeCursor :38`) — traversée
  **arête par arête**
- `solution : AgentSolution` (`97`), `local_memory : DeliveryLocalMemory` (`98`)

`AgentSolution` (`AgentSolution.hpp:29`) : `current_position : const id*` (`30`),
`sequence : vector<SolutionStep>` (`31`) ; `SolutionStep{node, estimated_arrival}`
(`:12`).

`DeliveryLocalMemory` (`DeliveryAgent.hpp:73`) :
- `tasks : vector<PDPTask*>` (`74`)
- `current_path` / `next_path : const ObjectivePath*` (`75-76`) — **dual-path
  lookahead** (jambe courante + suivante)
- `operable_env : OperableEnvironment` (`78`), `local_agents` (`79`)
- `reroute_path : ObjectivePath` (`84`) — propriétaire stable d'un reroute TD-A*
  poussé par le Manager (le cache ne stocke que des chemins statiques)

**Module planning** : `receive_task` → DbVNS (legacy) sur `OperableEnvironment`
(`OperableEnvironment.hpp:20`), volontairement léger :
`nodes : vector<ObjectiveNode>` (`21`), `costs_ : vector<float>` plat N×N
(`61`), `index_map_ : unordered_map<id,int>` (`63`). Le contexte temporel
(`set_time_context` / `get_cost_at` → `bpr_path_cost`, mémoïsé dans
`tcost_memo_`, `:69`) recoûte chaque chemin à l'instant **`t*`** de la
recherche (pas à `t` figé de l'action).

**Module scoring** : `bid_for_task` (`DeliveryAgent.cpp`) construit le vecteur
12-d (`PolicyFeatures`, `PolicyKit.hpp:42`) à partir de ses tâches, son plan,
et le dual-path ; renvoie μ via la policy active.

### 3.3 Tâche — `src/DMASforPD/Utility/PDPTask.hpp`

`ObjectiveNode` (`ObjectiveNode.hpp:9`) : `id`, `group_id`.
`PDPTask` (`:38`) : `task_id`, `pickup`/`delivery : ObjectiveNode` (`40-41`),
`status : TaskStatus` (`42` / enum `:11`), `agent_id` (`43`),
`reward`/`importance` (`48-49`) + `*_original` immuables (`55-56`),
`timeline : TaskTimeline` (`58` / `:23`).

---

## 4. Policies — `src/DMASforPD/Policy/`

Socle `PolicyKit` (`PolicyKit.{hpp,cpp}`) : `Mlp 12→64→64→1`
(`Mlp`, `PolicyKit.hpp:150`), PPO complet dans `ppo_train`
(`PolicyKit.cpp:197`) : clip, value clip, KL early-stop, value-norm
(`RunningMeanStd`, `:86`), GAE par trajectoire (`compute_gae`, `:184`), Adam
(`MlpAdam`, `:185`), annealing lr/entropy (`PPOParams::set_progress`, `:127`).
Interface commune `IBidPolicy` (`BidPolicy.hpp:23`) : buffers par agent
(`buffers_`), plumbing de récompense, `recent_records_` pour apparier candidats
↔ entrées de buffer.

| Policy | Article | Critic input | Init / hyperparams | Réf |
|--------|---------|--------------|--------------------|-----|
| MAPPO | Yu+2022 | global 32-d (état 20-d ⊕ obs 12-d) | orthogonal, clip 0.1, 10 ep | `MappoPolicy.cpp` |
| IPPO | de Witt+2020 | local 12-d | He, clip 0.2, 4 ep, lr 1e-4 | `IPPOPolicy.cpp` |
| MAPPER | Liu+2020 | local 12-d, **par agent** | + évolution tous les 5 rounds | `MapperPolicy.cpp` |
| Hybrid | (aucun) | base MAPPO gelée + résiduel | REINFORCE en ligne, bornes/rollback | `HybridPolicy.cpp` |
| RMCA | Chen+2021 | — (non appris) | score = transform du coût marginal | `DeliveryAgent.cpp` |

**Mécanismes** : l'acteur sort μ = P(accepter). En training l'action est
échantillonnée `Bernoulli(μ)` (PPO exige a ~ π), en éval c'est `μ ≥ 0.5`. Le
critic estime `V(s)` (baseline de variance), GAE donne l'avantage, la perte
clippée borne le pas de politique. MAPPO = CTDE (critic global au training,
acteur local en exécution). IPPO teste si un critic local suffit ; MAPPER
décentralise tout + sélection évolutionnaire
`p_i = 1 − exp(η(R̄_i − R̄_best))` (copie exacte du meilleur, sans mutation,
`MapperPolicy.cpp:98`) ; Hybrid spécialise en ligne une base partagée gelée
`μ = σ(z_base + w_i·x + b_i)` (`HybridPolicy.cpp`).

---

## 5. Scénarios — `make_scenario_grid` (`EpisodeRunner.cpp`)

3 niveaux de tâche × 3 niveaux de congestion = **9 combinaisons déterministes**
(tables éditables en un point) :
- Tâche : easy `(density 0.7, agents ×1.2)` / normal `(1.0, ×1.0)` /
  hard `(1.8, ×0.7)`.
- Congestion : easy `(Flat, 1/way)` / normal `(Wave, 2)` / hard `(ShockBurst, 4)`.

`EpisodeScenario` (`EpisodeRunner.hpp`) porte `density_mult`, `agents_mult`,
`congestion_profile`, `ghost_density_per_hot_way` (>0 surcharge l'`EpisodeConfig`).
**Training** : le trainer parcourt la grille en boucle déterministe sur les
slots (round, ville) (`MultiCityTrainer.cpp`). **Éval** : méthode × ville × 9
scénarios, **1 épisode** chacun. Paramétrable depuis `main`
(`cfg.eval_scenarios` / `cfg.train_scenarios`).

---

## 6. SoTA (niveaux de comparaison)

- **RMCA** (policy) : remplace le scoring RL dans le pipeline TAM+DbVNS.
- **Token Passing** (allocation) : remplace **TAM + policy** par argmin
  `h(loc, pickup)` sur agents libres (Ma+2017), branche dédiée dans
  `offer_task` (`EpisodeRunner.cpp:1340`).
- **CA** (`FaithfulCASolver`) et **HAPC** (`HybridAdaptivePredictiveSolver`) :
  pipelines complets autonomes via `SolverRunner` (`SoTA/`), fidèles à l'article
  (CA : A* BPR + γ-mode + tie-break β_W ; HAPC : dispatch prédictif 2-pas,
  objectif eq.11, énumération exhaustive au lieu du PSO). Évalués sur le même
  `SharedEpisodeSetup` (mêmes tâches, positions, ghost s).

---

## 7. Workflow runtime (un épisode — `EpisodeRunner::run`)

1. **Reset** : `reset_episode` + reset des agents + clear des buffers policy ;
   publication policy active / stratégie planning / speed.
2. Boucle `step = 0..T` :
   - `advance_time(step)` (horloge + purge congestion passée)
   - `ghost_traffic_.step(step)` (trafic exogène)
   - `build_global_state` → `memory_.cur_global_state`
   - **Arrivées** : `add_task` (crée le `PDPTask`, l'inscrit dans `available_` +
     `node_to_task_id_`) puis `offer_task` :
     - baselines (Greedy/Random/InsertionGreedy/PIBT/TP/MCA/DbVNS/ALNS) :
       sélection directe, sans buffer RL ;
     - RL/RMCA/TamAlwaysAccept : `on_new_task` crée le task agent, TAM `step()`
       jusqu'à allouer/épuiser ; chaque bid enregistre une expérience.
     - sur allocation : `commit_accepted_task` → si idle, `start_leg`.
   - **Mouvement** : `process_arrivals` → `on_objective_reached` (pickup/delivery)
     replanifie : `begin_leg` → `commit_plan` → `schedule_next_edge`
     (commit **avant** schedule, cf. §3.2 self-exclusion).
   - Récompenses : crédit pickup à l'arrivée pickup, crédit delivery (×facteur
     latence) à la livraison ; pénalités non-affecté / unfinished / idle.
3. **Fin** : pénalités unfinished/idle ; `train_round()` de la policy active
   (ou online pour Hybrid) ; calcul des métriques.

**Cycle de vie d'une tâche** : `add_task` (available) → TAM `assign_task`
(allocated) → `receive_task` (insertion plan) → mouvement → pickup → delivery →
`complete_task` (finished), node objectif retiré du cache (`clear_objective`).
**Cycle de vie d'un agent** : créés une fois par l'EpisodeRunner, réutilisés ;
remis à zéro à chaque `run()` ; Idle → reçoit tâche → Active → vide sa file →
Idle.

---

## 8. Règle de traversée (point critique, §1.2 + §3.2)

Un agent entrant sur une arête à `t` met le temps BPR évalué sur les `n`
**autres** présents, pas `n+1` : il n'est affecté ni par son propre poids ni par
ceux qui entrent après. Implémentation : `schedule_next_edge`
(`EpisodeRunner.cpp:1690`) appelle `traversal_steps(..., self_w)` avec
`self_w = load_per_agent` ; le `commit_plan` précède toujours le
`schedule_next_edge` pour que la charge propre soit présente dans `load_` au
moment de la soustraction. Les requêtes de **planning/prédiction** passent
`self_weight = 0` (champ complet).

---

## 9. Points ouverts

- **TP fidèle (agent-pull)** : aujourd'hui *task-centric* (la tâche choisit
  l'agent libre le plus proche). Version Ma+2017 stricte = pool de tâches non
  allouées (`available_tasks`) + chaque agent libre (ordre fixe) tire
  `argmin_τ h(loc, s_τ)` ; déplace l'allocation de `offer_task` vers la boucle.
- Découpage éventuel d'`EpisodeRunner` (simulation / métriques).
- Self-exclusion du poids propre dans le mouvement interne des solveurs
  autonomes (CA/HAPC) pour cohérence totale.
