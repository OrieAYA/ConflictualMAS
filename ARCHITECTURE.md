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

**Ghosts mobiles — `MovingGhosts.{hpp,cpp}` (module écrit, NON BRANCHÉ)** :
`MovingGhostTraffic` remplace les événements statiques par de vraies
trajectoires — chaque ghost spawn (densité ∝ profil temporel, biais hot-way),
marche le graphe arête par arête à vitesse free-flow, puis sort. Le jeu de
`GhostTransit{ghost_id, edge, t_entry, t_exit, load}` est une fonction pure
de la seed (replays identiques), mais un transit n'est **observable qu'au
moment où il a lieu** :
- `reveal_to_map=true` : `step(t, cmap)` pousse chaque transit dans la map à
  son `t_entry` (online en temps, global en espace — le futur d'un ghost
  n'est jamais visible du Manager, contrairement au contrôleur actuel qui
  injecte tout au reset).
- `reveal_to_map=false` (mode online post-éval) : rien n'est écrit ; la
  vérité vit dans le module (`truth_load(edge,t)`, `observe_incident(node,t)`)
  et le Manager ne connaît que ce que la flotte a vu.

Plan de branchement (détaillé en tête de `MovingGhosts.hpp`) :
(1) mode révélé : `setup_ghost_traffic` instancie sur `cfg.ghost_moving`
(flags `ghost_moving`/`ghost_move_*` déjà dans EpisodeConfig, OFF), la boucle
remplace `ghost_traffic_.step(step)` par `step(step, congestion_map)` ;
parité de volume `n_ghosts ≈ n_events·window/durée_route_moyenne`.
(2) mode online : `reveal_to_map=false` + physique du mouvement ajoute
`truth_load` au coût de traversée (`schedule_next_edge`) + à chaque
`arrive_at_node` le runner ingère `observe_incident` dans la map en ghost
loads courte durée (`add_ghost_load(edge, t, t+ttl, load)`).
(3) LSM : l'entrée g[4] passe au comptage d'arêtes fraîchement observées.

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
- `geo_box&`, `server_memory` (path caches), `congestion_map`, `task_agent`
  (les recherches statiques libres — A*, Dijkstra, haversine — vivent dans
  `Environment/GeoBox/GraphSearch.hpp`, namespace `graph_search`)
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
`paths_` est conservée **entre replays du même épisode** (même `ep_seed` :
les 3 policies d'un point de grille partagent le cache) ; seuls
`dynamic_cost`/`dynamic_step` sont invalidés (`ObjectiveGroupCache::
episode_reset`). Quand le contenu d'épisode change (`ep_seed` différent),
`prepare_run` purge tout le cache (`PDPServerMemory::clear_paths`) — sans
cette borne il croît sans limite sur une grille (~24 Go observés, 2 crashs
mémoire). Bornes complémentaires : les états Dijkstra incrémentaux d'une
tâche sont libérés dès la destruction de son TAM (`offer_task`,
RunnerAllocation.cpp) ; le trainer appelle `release_episode_memory()` après
chaque point de grille (les runners inactifs ne retiennent rien) ; la ligne
de log d'épisode affiche `mem=` (commit du process, Mo). Rejouer un même scénario sur une autre méthode ne superpose donc
aucun effet.

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
5. Zéro candidat après épuisement des deux Dijkstra → tâche abandonnée,
   comptée `n_no_candidate` (échec capacité/topologie, **≠ refus de policy**).
   Métriques : `accept_rate = accepté/(accepté+refusé)` ≡ 1.0 en Format A ;
   `tasks_appeared = accepté + refusé + no_candidate` (throughput inchangé).
   Les features du GlobalState gardent `no_candidate` dans leur dénominateur
   (parité avec les checkpoints entraînés).

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
- `plan_cong : unordered_map<way_id,float>` — congestion normalisée de chaque
  arête committée **au moment du planning** (snapshot par
  `register_committed_plan` quand `record_plan_congestion` ; feature f2 §4.1)

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

### 4.1 Movement Decision Policy — `MovementPolicy.{hpp,cpp}`

Gate PPO appris sur le replanning local (slide 36 du plan de recherche) :
**une décision par arrivée sur node** — faut-il recalculer la jambe courante
via TD-A* ? Mono-agent : un seul actor/critic partagé par la flotte,
totalement découplé des bid policies (structures propres `MoveObs` /
`MoveExperience`, PPO recopié de `ppo_train` — `Experience`/`kPolicySz`
intacts).

- **Observation** (`kMoveObsSz = 36`) : 4 scalaires — congestion de l'arête
  planifiée suivante maintenant (`f1`), la même au moment du planning (`f2`,
  snapshot `plan_cong`), min des autres arêtes incidentes hors arête
  d'arrivée (`f3`, 1 = pas d'alternative), alerte LSM de l'agent (`f4`,
  §4.2, 0 si LSM off) — + étoile locale : K=8 slots d'arêtes incidentes × 4
  features `[cong_now, plan_cong, is_next, is_from]`.
  Normalisation `c = x/(1+x)`, `x = get_load/edge_capacity`.
- **Réseau** `StarNet` : encodeur partagé par arête `16×4` → ReLU → pooling
  mean‖max (32) → concat scalaires (35) → tête `Mlp{35→64→64→1}`
  (invariance par permutation, degré variable). Backward encodeur manuel
  (`StarNet::backprop`), Adam via `policy_optim`. Hyperparams colonne IPPO
  (clip 0.2, 4 epochs, lr 1e-4, grad 0.5, ent 5e-3).
- **Reward mixte** : à la décision, si replan : `g − κ − β·1[g<ε]` avec
  `g = max(0, cur−tda)/cur` mesuré par `RerouteOutcome` (κ=`mp_replan_cost`,
  β=`mp_null_gain_penalty`, ε=`mp_gain_eps`) ; à l'objectif de la jambe,
  chaque décision ouverte reçoit `w_out·clamp((pred−actual)/pred, −1, 1)`
  (`credit_leg_outcome`, pred = `estimated_arrival − step` au moment de la
  décision).
- **Flags** (`EpisodeConfig`, OFF par défaut = comportement legacy exact) :
  `use_movement_policy` (le gate remplace le check >5% inconditionnel aux
  objectifs et s'applique aussi aux nodes intermédiaires),
  `movement_train` (échantillonnage Bernoulli + `train_round` fin d'épisode),
  `mp_*`.
- **Branchement** : `process_arrivals` (node intermédiaire,
  `RunnerMovement.cpp`) et les deux frontières d'objectif
  (`on_objective_reached`) appellent `movement_decision` → si oui,
  `push_rerouted_path(agent, speed, &out)` (TD-A* + règle >5% inchangée).
  Snapshot `plan_cong` rempli par `register_committed_plan` quand
  `PDPGlobalMemory::record_plan_congestion` (posé par `prepare_run`).
- **Checkpoint** : magic `0xDEA110D2`, actor+critic+vrms (format MAPPO).

### 4.2 LSM Prediction/Signaling — `src/DMASforPD/Prediction/Lsm.{hpp,cpp}`

Liquid State Machine (Maass) côté serveur (slide 37 Mechanisms ; papier §7
« refinement of congestion level prediction ») : prédit la congestion à
t+H par zone et **signale les agents** dont la route planifiée traverse une
zone d'alerte — le signal alimente la feature f4 de la movement policy
(pattern slide 33 : serveur → signal → truck → policy confirme).

- **Liquide fixe** (`LsmModule`, singleton `lsm_module()`) : 256 LIF sur
  lattice 4×4×16, 80/20 exc/inh, connectivité `p = C·exp(−d²/λ²)` (C Maass
  2002 par type de paire), lignes normalisées (Σ|w| = gain). Tick :
  `v = 0.9·v + W_in·u + W_res·s` ; spike si v>1 (reset) ; état liquide =
  trace exponentielle normalisée des spikes.
- **Entrée u (136)** : 64 congestion coarse 8×8 (agrégation des
  `cell_cong_cache` 32×32 de la RegionStatsGrid, normalisation absolue
  `(mul−1)/mul`) + 64 densité coarse (`task_counts/max_count`) + 8 globaux
  (time_ratio, mean/peak load, n_edges_load_ge(2), ghosts actifs, λ
  arrivées, ratio agents actifs, ratio tâches disponibles).
- **Sortie ŷ (65)** : congestion coarse prédite à t+H (64) + niveau global.
  Seul le readout `W_out` apprend : **NLMS online** avec supervision différée
  (ring buffer d'états ; à t+H la valeur réalisée devient la cible).
- **Branchement** (`EpisodeRunner::lsm_tick`, `RunnerPrediction.cpp`) :
  tick tous les `lsm_every` steps dans la boucle (avant `process_arrivals`) ;
  zones d'alerte = `ŷ ≥ lsm_alert_threshold` ; alerte agent = max ŷ des
  cellules croisées par sa route restante (cursor, stride ≤16 lookups) +
  `next_path` → `lsm_alert_` lu par `movement_decision` (f4, 0 si absent).
- **Flags** (`EpisodeConfig`, OFF par défaut = zéro impact) : `use_lsm`,
  `lsm_train`, `lsm_every=10`, `lsm_horizon=100`, `lsm_alert_threshold=0.5`,
  `lsm_lr=0.1`. Checkpoint : magic `0xDEA110D4` (W_in, W_res, W_out).

---

## 5. Scénarios, training et évaluation

### 5.1 Grille de scénarios — `make_scenario_grid` (`ScenarioConfig.cpp`)

3 profils temporels de tâche (`Uniform`/`ShockPick`/`Wave`) × 3 profils de
congestion (mêmes formes) × 3 régimes de flotte (AM = 0.7/1.0/2.5) =
**27 combinaisons déterministes** (tables éditables en un point :
`paper_task_regimes` / `paper_congestion_regimes` / `paper_fleet_regimes`).
`EpisodeScenario` (`ScenarioConfig.hpp`) porte `task_profile`,
`congestion_profile`, `agents_mult` ; les quantités d'événements sont fixées
par `event_tuning` (tasks = round(100·SCE·RM), ghosts = round(25000·SCE·RM),
flotte = round(10·SCE·AM)).

### 5.2 Training — `MultiCityTrainer::train_grid` (`Trainer.cpp`)

Grille : 6 villes (3 villes × 2 tailles) × 3 seeds × 27 scénarios =
**486 épisodes / policy** (162 par seed). Par seed, chaque policy (MAPPO,
IPPO, MAPPER) est **ré-initialisée** puis apprend sur les 162 épisodes de la
seed : un point de grille = **un épisode généré une seule fois**
(`build_shared_episode_setup` : stream de tâches, positions de départ,
capacités hétérogènes, seed ghost) **rejoué par les 3 policies** — comparaison
équitable et pas de régénération. Progression (annealing lr/entropy et poids
de shaping §9) indexée sur l'épisode dans la seed. Checkpoints
`{out}/{mappo,ippo,mapper}/{policy,ippo,mapper}_seed{seed}.bin` **réécrits
après chaque épisode** (perte max = 1 épisode) ; CSV `episodes_train.csv`
flushé au même rythme, `summary.csv` par seed.

**Training movement** — `MultiCityTrainer::train_movement` (`Trainer.cpp`,
menu `5`) : côté bid **gelé** (checkpoint MAPPO `cfg.policy_path` chargé,
`train_mode=false` → pas d'exploration ni de train_round bid) ; seule la
movement policy (§4.1) apprend (`use_movement_policy` + `movement_train`
posés par main.cpp). Grille réduite par défaut (Tokyo Small/Medium × 27
scénarios, 1 seed). Sorties : `{out}/movement/movement_seed{seed}.bin`
(réécrit chaque épisode) + `movement_train.csv` (décisions, replans,
adoptions, gain moyen, stats PPO par épisode). Pas de resume (v1).
Si `results/lsm_train/lsm/lsm_seed42.bin` existe, main.cpp l'active
(`cfg.lsm_path`, `use_lsm=true`, `lsm_train=false`) → LSM figé, f4 vivante.

**Pretraining LSM** — `MultiCityTrainer::pretrain_lsm` (`Trainer.cpp`,
menu `6`) : bid gelé (MAPPO éval), movement policy OFF, `use_lsm` +
`lsm_train` → le readout NLMS apprend online. Sorties :
`{out}/lsm/lsm_seed{seed}.bin` (réécrit chaque épisode) + `lsm_train.csv`
(mse, cellules/agents en alerte, throughput par épisode).

**Reprise après crash** (`cfg.resume`, ON par défaut ; smoke : OFF) :
`completed_grid_points` relit `episodes_train.csv` (un point est complet si
toutes les policies sélectionnées l'ont loggé — flush par point) et **purge
du fichier les lignes des points incomplets** (crash en cours d'écriture) :
le CSV final reste un dataset unique et propre. Seeds complètes sautées ; seed partielle : `reinit` puis
`load` des checkpoints `_seed{seed}.bin`, skip des `done` premiers points
(les ep_seeds sont des fonctions pures de (seed, ci, sc) → contenu identique),
CSV rouvert en **append**. Perdus à la reprise : moments Adam (non persistés)
et compteur d'évolution MAPPER — impact borné à un point de grille. La
reprise suppose une grille inchangée (mêmes villes/scénarios/modes) ; le
`summary.csv` d'une seed reprise ne couvre que la portion post-reprise.

### 5.3 Évaluation — `MultiCityTrainer::evaluate` (`Trainer.cpp`) + Phase B

`evaluate` charge les checkpoints par seed (`policy_path` /
`ippo_policy_path` / `mapper_policy_path` — un mode RL demandé sans chemin =
abort) puis lance `run_eval`, **épisode-majeur** : pour chaque slot (ville,
scénario, épisode), un seed déterministe → **un** `SharedEpisodeSetup`
construit une seule fois (l'état initial), rejoué par chaque mode de
`eval_modes` (`prepare_run` restaure l'état initial et vide les buffers de
la méthode précédente), puis par les solveurs standalone **CA + HAPC**
(`SolverRunner` frais sur le même setup) — environnements byte-identiques,
jointure CSV sur (city, scenario, episode). Les caches de chemins du slot
sont partagés entre méthodes (données statiques déterministes) et libérés en
fin de slot (`release_episode_memory`). Sorties : `episodes_seed{seed}.csv`
(modes) + `sota_standalone/sota_seed{seed}.csv` (CA/HAPC). Hybrid se
construit à l'évaluation : base = actor MAPPO chargé, résiduels en ligne.
Les anciennes phases stress/généralisation et le flag
`use_shared_episode_setup` ont été supprimés (protocole = sweep RM,
15 villes, toutes en phase unique).

**Protocole option 4 (main) — sweep de charge parallélisable** : villes
Small/Medium des 5 familles (SCE 1/2 ; les Larges SCE 3 existent au registre
mais sont hors protocole), grille des 27 scénarios, 6 modes pipeline (MAPPO,
IPPO, MAPPER, Hybrid, RMCA, TokenPassing — TamAlwaysAccept retiré :
redondant en Format A, conservé dans le code pour les tests) + CA/HAPC
standalone. Au lancement, saisie du **groupe de villes** (1 = Tokyo+Kyoto,
2 = LosAngeles+NewYork, 3 = Paris, 0 = toutes) et de la **plage RM** — un
terminal par groupe = 3 process parallèles sans conflit. RM avance de 0,5 ;
`seed = 42 + (RM−1)/0,5` (fonction de RM → un niveau produit les mêmes
épisodes quel que soit le découpage) ; `ratio_mult = RM` (tasks =
100·SCE·RM, ghosts = 25000·SCE·RM, flotte inchangée). Checkpoints
d'entraînement **fixes** (`kEvalPolicySeed`). Sorties :
`results/paper_eval/pol{P}_rm{RM}_g{groupe}/{episodes_seed{seed}.csv,
sota_standalone/sota_seed{seed}.csv}` — schémas identiques entre groupes,
consolidation par concaténation.

---

## 6. SoTA (niveaux de comparaison)

- **RMCA** (policy) : remplace le scoring RL dans le pipeline TAM+DbVNS.
- **Token Passing** (allocation) : remplace **TAM + policy** par argmin
  `h(loc, pickup)` sur agents libres (Ma+2017), branche dédiée dans
  `offer_task` (`RunnerAllocation.cpp`).
- **ALNS / Double-Horizon** (planning, Table 8 de l'article) : allocation =
  argmin du coût marginal d'insertion, planner sélectionné via les flags
  `planning_use_*` (`prepare_run`). DbVNS est le planner du framework.
- **CA** (`FaithfulCASolver`) et **HAPC** (`HybridAdaptivePredictiveSolver`) :
  pipelines complets autonomes via `SolverRunner` (`SoTA/`), fidèles à l'article
  (CA : A* BPR + γ-mode + tie-break β_W ; HAPC : dispatch prédictif 2-pas,
  objectif eq.11, énumération exhaustive au lieu du PSO). Évalués sur le même
  `SharedEpisodeSetup` (mêmes tâches, positions, ghost s).

---

## 7. Workflow runtime (un épisode — `EpisodeRunner::run`)

Implémentation découpée en trois fichiers (`src/TrainingEvaluation/Run/`) :
`Runner.cpp` (cycle de vie : prepare, boucle, récompenses de fin),
`RunnerAllocation.cpp` (protocole d'offre + coûts d'insertion),
`RunnerMovement.cpp` (mouvement arête par arête + arrivées).

1. **Reset** : `reset_episode` + reset des agents + clear des buffers policy ;
   publication policy active / stratégie planning / speed.
2. Boucle `step = 0..T` :
   - `advance_time(step)` (horloge + purge congestion passée)
   - `ghost_traffic_.step(step)` (trafic exogène)
   - `lsm_tick(step, …)` si `use_lsm` et step multiple de `lsm_every`
     (§4.2 : tick liquide, apprentissage différé, zones + alertes agents)
   - `build_global_state` → `memory_.cur_global_state`
   - **Arrivées** : `add_task` (crée le `PDPTask`, l'inscrit dans `available_` +
     `node_to_task_id_`) puis `offer_task` :
     - baselines (Greedy/Random/InsertionGreedy/TP/DbVNS/ALNS/DH) :
       sélection directe, sans buffer RL ;
     - RL/RMCA/TamAlwaysAccept : `on_new_task` crée le task agent, TAM `step()`
       jusqu'à allouer/épuiser ; chaque bid enregistre une expérience.
     - sur allocation : `commit_accepted_task` → si idle, `start_leg`.
   - **Mouvement** : `process_arrivals` → `on_objective_reached` (pickup/delivery)
     replanifie : `begin_leg` → `commit_plan` → `schedule_next_edge`
     (commit **avant** schedule, cf. §3.2 self-exclusion). Avec
     `use_movement_policy`, chaque arrivée sur node (intermédiaire ou
     objectif) passe par `movement_decision` (§4.1) qui gate l'appel
     `push_rerouted_path` ; OFF = check >5% inconditionnel aux objectifs
     seulement (comportement historique).
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
(`RunnerMovement.cpp`) appelle `traversal_steps(..., self_w)` avec
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
- Self-exclusion du poids propre dans le mouvement interne des solveurs
  autonomes (CA/HAPC) pour cohérence totale.
