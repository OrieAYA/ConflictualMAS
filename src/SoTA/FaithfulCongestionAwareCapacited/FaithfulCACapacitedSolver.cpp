#include "FaithfulCACapacitedSolver.hpp"
#include "Environment/Congestion/CongestionMap.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

void FaithfulCACapacitedSolver::init(const SolverContext& ctx) {
    ctx_ = &ctx;

    agents_.clear();
    agents_.reserve(static_cast<size_t>(ctx.n_active_agents));
    for (int i = 0; i < ctx.n_active_agents; ++i) {
        AgentState a;
        a.current_node = (i < static_cast<int>(ctx.agent_start_nodes.size()))
                       ? ctx.agent_start_nodes[i]
                       : 0;
        a.capacity = (i < static_cast<int>(ctx.per_agent_capacity.size()) &&
                      !ctx.per_agent_capacity.empty())
                   ? ctx.per_agent_capacity[i]
                   : std::max(1, ctx.max_capacity_per_agent);
        agents_.push_back(std::move(a));
    }

    pending_task_ids_.clear();
    tasks_.clear();
    appeared_ = completed_ = refused_ = 0;
    latency_sum_ = wait_sum_ = trip_sum_ = 0;
    road_pd_sum_ = 0.0;
    road_pd_count_ = 0;
    active_steps_sum_ = 0;
    capacity_violations_ = pairing_violations_ = 0;
    instr_.init(ctx.n_active_agents);
}

void FaithfulCACapacitedSolver::inject_task(const ScheduledTask& task, int step) {
    TaskRecord r;
    r.task_id        = static_cast<int>(tasks_.size());
    r.pickup_node    = task.pickup_node_id;
    r.delivery_node  = task.delivery_node_id;
    r.arrival_step   = step;
    // Pre-compute the FREE-FLOW-ish pickup→delivery road distance for the
    // route_efficiency metric (mirrors FaithfulCASolver verbatim).
    BPRPath warm = bpr_a_star(r.pickup_node, r.delivery_node, step);
    if (warm.valid) {
        const auto& ways = ctx_->geo_box->data.ways;
        for (auto eid : warm.edges) {
            auto it = ways.find(eid);
            if (it != ways.end()) r.pd_road_dist += it->second.distance_meters;
        }
    }
    tasks_.push_back(r);
    pending_task_ids_.push_back(r.task_id);
    ++appeared_;
}

int FaithfulCACapacitedSolver::edge_arrival_step(osmium::object_id_type edge_id,
                                                  int t_enter) {
    if (!ctx_) return t_enter + 1;
    const auto& ways = ctx_->geo_box->data.ways;
    auto it = ways.find(edge_id);
    if (it == ways.end()) return t_enter + 1;
    const float length_m  = it->second.distance_meters;
    const float base_time = length_m / std::max(0.1f, ctx_->speed_mps);
    const float adj = ctx_->congestion_map
        ? ctx_->congestion_map->adjusted_cost(edge_id, base_time, length_m, t_enter)
        : base_time;
    // BPR factor sampled at the edge's ENTRY step (t_enter = current step at all
    // call sites). Must be here, not at completion: advance() purges past steps
    // → a later query at the entry step reads load 0 → BPR 1.0.
    if (ctx_->congestion_map && base_time > 0.f)
        instr_.record_edge_bpr(adj / base_time);
    return t_enter + std::max(1, static_cast<int>(std::ceil(adj)));
}

// ── BPR-aware A* ─────────────────────────────────────────────────────────────
// Mirrors FaithfulCASolver::bpr_a_star verbatim — same admissible heuristic
// (euclidean / speed), same OpenEntry priority queue, same edge-cost lookup
// against the CongestionMap at the predicted arrival step at the edge head.
// Re-implemented privately here so this solver remains self-contained.
FaithfulCACapacitedSolver::BPRPath
FaithfulCACapacitedSolver::bpr_a_star(osmium::object_id_type from,
                                       osmium::object_id_type to,
                                       int start_step) const {
    BPRPath result;
    if (!ctx_ || !ctx_->geo_box || !ctx_->pathfinder) return result;
    if (from == 0 || to == 0) return result;
    if (from == to) {
        result.valid = true;
        result.nodes = { from };
        result.trip_time = 0.f;
        return result;
    }

    const auto& nodes = ctx_->geo_box->data.nodes;
    const auto& ways  = ctx_->geo_box->data.ways;
    const float speed = std::max(0.1f, ctx_->speed_mps);

    auto end_it = nodes.find(to);
    if (end_it == nodes.end()) return result;

    auto h_to_goal = [&](osmium::object_id_type n) -> float {
        const float h_m = const_cast<Pathfinder*>(ctx_->pathfinder)
                              ->heuristic(n, to);
        return h_m / speed;
    };

    struct OpenEntry {
        float f;
        float g;
        int   t_arrive;
        osmium::object_id_type node;
        bool operator>(const OpenEntry& o) const { return f > o.f; }
    };

    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;
    std::unordered_map<osmium::object_id_type, float> g_score;
    std::unordered_map<osmium::object_id_type,
        std::pair<osmium::object_id_type, osmium::object_id_type>> came_from;
    std::unordered_set<osmium::object_id_type> closed;

    g_score[from] = 0.f;
    open.push({ h_to_goal(from), 0.f, start_step, from });

    while (!open.empty()) {
        OpenEntry cur = open.top();
        open.pop();
        if (closed.count(cur.node)) continue;
        closed.insert(cur.node);

        if (cur.node == to) {
            std::vector<osmium::object_id_type> rev_nodes;
            std::vector<osmium::object_id_type> rev_edges;
            osmium::object_id_type cn = to;
            rev_nodes.push_back(cn);
            while (cn != from) {
                auto it = came_from.find(cn);
                if (it == came_from.end()) break;
                rev_edges.push_back(it->second.second);
                cn = it->second.first;
                rev_nodes.push_back(cn);
            }
            std::reverse(rev_nodes.begin(), rev_nodes.end());
            std::reverse(rev_edges.begin(), rev_edges.end());
            result.nodes = std::move(rev_nodes);
            result.edges = std::move(rev_edges);
            result.trip_time = cur.g;
            result.valid = true;
            return result;
        }

        auto node_it = nodes.find(cur.node);
        if (node_it == nodes.end()) continue;
        const auto& pt = node_it->second;

        for (auto edge_id : pt.incident_ways) {
            auto wit = ways.find(edge_id);
            if (wit == ways.end()) continue;
            const auto& w = wit->second;
            const osmium::object_id_type nbr =
                (w.node1_id == cur.node) ? w.node2_id : w.node1_id;
            if (nbr == 0) continue;
            if (closed.count(nbr)) continue;

            const float length_m = w.distance_meters;
            if (length_m <= 0.f) continue;
            const float base_time = length_m / speed;
            const float adj = ctx_->congestion_map
                ? ctx_->congestion_map->adjusted_cost(edge_id, base_time,
                                                      length_m, cur.t_arrive)
                : base_time;
            const float tentative_g = cur.g + adj;

            auto gs = g_score.find(nbr);
            if (gs != g_score.end() && tentative_g >= gs->second) continue;

            g_score[nbr] = tentative_g;
            came_from[nbr] = { cur.node, edge_id };
            const int new_t = cur.t_arrive + std::max(1, static_cast<int>(std::ceil(adj)));
            const float f = tentative_g + h_to_goal(nbr);
            open.push({ f, tentative_g, new_t, nbr });
        }
    }

    return result;
}

// ── Memoised BPR-A* lookup ──────────────────────────────────────────────────
// Wraps bpr_a_star with a per-step cache keyed by (from, to). The cache is
// flushed at the start of every step(), so the snapshot is always consistent
// with the CongestionMap at the current allocation timestep.
FaithfulCACapacitedSolver::BPRPath
FaithfulCACapacitedSolver::bpr_lookup(osmium::object_id_type from,
                                      osmium::object_id_type to,
                                      int step) const
{
    if (cache_valid_at_step_ != step) {
        bpr_cache_.clear();
        cache_valid_at_step_ = step;
    }
    NodePairKey key;
    key.from = from;
    key.to   = to;
    auto it = bpr_cache_.find(key);
    if (it != bpr_cache_.end()) return it->second;
    BPRPath p = bpr_a_star(from, to, step);
    bpr_cache_.emplace(key, p);
    return p;
}

// ── Total BPR-time of an ordered sequence ───────────────────────────────────
// Walks the sequence start→stop, summing BPR-A* trip_time of each leg under
// the CURRENT-STEP congestion snapshot. We deliberately do NOT propagate
// start_step through the legs as the original FaithfulCA::decision_cost does
// for its 2-leg case — propagation would defeat the per-step bpr_lookup cache
// (each leg would query a different start_step). The fidelity cost is small:
// the CongestionMap's BPR factor is slow-changing within the small time
// window of a single sequence (~tens of seconds), and Asadi+2025's §3.2
// formulation evaluates the modified A* "given current estimated congestion"
// at the moment of planning — a snapshot, not a propagation. With the cache,
// per-allocation cost drops from O(N_agents × n³) BPR-A* down to O(unique
// (from, to) pairs) ≈ O(N_agents × n) — comparable to FaithfulCA itself.
float FaithfulCACapacitedSolver::sequence_bpr_time(
    osmium::object_id_type from_node,
    const std::vector<SequenceStop>& seq,
    int start_step) const
{
    if (seq.empty()) return 0.f;
    float total = 0.f;
    osmium::object_id_type cur = from_node;
    for (const auto& s : seq) {
        BPRPath leg = bpr_lookup(cur, s.node, start_step);
        if (!leg.valid) return std::numeric_limits<float>::max();
        total += leg.trip_time;
        cur = s.node;
    }
    return total;
}

// ── Capacitated cheapest insertion ──────────────────────────────────────────
// Structural skeleton borrowed from HAPC (paper §3.3 EE over (pos_p, pos_d)
// with capacity-feasibility check on the load profile) but with cost =
// pure BPR-time sum — NOT HAPC's eq.11 (load × travel + α × waiting). This
// keeps FaithfulCA's cost philosophy intact while enabling multi-task
// insertion.
//
// Complexity: O(n²) candidate pairs × O(n) sequence BPR-time per candidate
// = O(n³) BPR-A* calls per (agent, task). Sequences are bounded by capacity
// (≤ 2K stops at any time → K ≤ 7 in our setup → n ≤ 14), so ~2700 BPR-A*
// per (agent, task) worst case. Acceptable for an event-triggered solver.
FaithfulCACapacitedSolver::InsertionResult
FaithfulCACapacitedSolver::cheapest_insertion(const AgentState& a,
                                               const TaskRecord& t,
                                               int step,
                                               float cutoff_raw_delta) const
{
    InsertionResult best;
    best.cost_delta = std::numeric_limits<float>::max();

    const int n = static_cast<int>(a.sequence.size());

    // Pre-compute the BASE sequence cost so each candidate's delta is just
    // (new_seq_cost − base_cost). This base cost is also the agent's
    // "current planned trip time" if no insertion happens.
    const float base_cost = sequence_bpr_time(a.current_node, a.sequence, step);
    if (!std::isfinite(base_cost)) return best;

    // Build the load profile of the base sequence to skip capacity-infeasible
    // candidates without rebuilding the full sequence each time.
    //   load_before[i] = #in-flight tasks BEFORE processing sequence[i]
    //   load_after[i]  = #in-flight tasks AFTER  processing sequence[i]
    std::vector<int> load_before(n);
    std::vector<int> load_after(n);
    int load = static_cast<int>(a.in_flight_task_ids.size());
    for (int i = 0; i < n; ++i) {
        load_before[i] = load;
        load += a.sequence[i].is_pickup ? +1 : -1;
        load_after[i] = load;
    }

    // Track the running best delta to enable early termination by lower
    // bound. Initialise from the caller's cutoff so we never explore beyond
    // what is useful to the outer (try_allocate_one) loop.
    float running_best = cutoff_raw_delta;

    std::vector<SequenceStop> candidate;
    candidate.reserve(static_cast<size_t>(n + 2));

    for (int pos_p = 0; pos_p <= n; ++pos_p) {
        // Capacity at the moment of pickup: pre-existing load_before[pos_p]
        // + 1 (the new pickup) ≤ capacity.
        const int load_at_pickup = (pos_p > 0) ? load_after[pos_p - 1]
                                                : static_cast<int>(a.in_flight_task_ids.size());
        if (load_at_pickup + 1 > a.capacity) continue;

        // ── Lower bound on delta for ANY (pos_p, *) ────────────────────────
        // Cheapest possible delta when inserting at pos_p, pos_d=pos_p (P+D
        // consecutive at pos_p, no detour through middle stops): equals the
        // BPR cost of {pred → pickup → delivery → next} minus the removed
        // edge {pred → next}. The pickup→delivery leg is fixed regardless of
        // pos_d, so it's the tightest LB available cheaply. If this LB
        // exceeds the running_best, no pos_d at this pos_p can beat it →
        // skip the whole inner loop.
        if (running_best < std::numeric_limits<float>::max()) {
            const osmium::object_id_type prev = (pos_p > 0)
                ? a.sequence[pos_p - 1].node : a.current_node;
            BPRPath leg_prev_pickup = bpr_lookup(prev, t.pickup_node, step);
            BPRPath leg_pickup_delivery = bpr_lookup(t.pickup_node, t.delivery_node, step);
            if (leg_prev_pickup.valid && leg_pickup_delivery.valid) {
                // SOUND lower bound on delta for ANY pos_d >= pos_p:
                //   delta >= leg(prev->P) + leg(P->D) - leg(prev->succ)
                // The cheapest realisation (P,D consecutive at pos_p) costs
                // leg(prev->P)+leg(P->D)+leg(D->succ) - leg(prev->succ); since
                // leg(D->succ) >= 0, dropping it yields a valid underestimate.
                //
                // BUGFIX: the previous version omitted the -leg(prev->succ)
                // term. That OVER-estimated the LB and could prune the optimal
                // near-zero-delta piggyback insertion — exactly the case this
                // solver exists to exploit. Subtracting the removed edge makes
                // the prune optimum-preserving (delta is never below this LB).
                float lb = leg_prev_pickup.trip_time + leg_pickup_delivery.trip_time;
                if (pos_p < n) {
                    BPRPath leg_prev_succ =
                        bpr_lookup(prev, a.sequence[pos_p].node, step);
                    if (leg_prev_succ.valid) lb -= leg_prev_succ.trip_time;
                }
                if (lb > running_best) continue;
            }
        }

        for (int pos_d = pos_p; pos_d <= n; ++pos_d) {
            // Capacity feasibility for ORIGINAL stops between pos_p and pos_d
            // — each gets +1 load weight because the new pickup is on board.
            bool feasible = true;
            for (int k = pos_p; k <= pos_d - 1; ++k) {
                if (load_after[k] + 1 > a.capacity) { feasible = false; break; }
            }
            if (!feasible) continue;

            // Materialise the candidate sequence: original prefix, P, original
            // middle, D, original suffix.
            candidate.clear();
            for (int i = 0; i < pos_p; ++i) candidate.push_back(a.sequence[i]);
            candidate.push_back({ t.task_id, t.pickup_node,   true  });
            for (int i = pos_p; i < pos_d; ++i) candidate.push_back(a.sequence[i]);
            candidate.push_back({ t.task_id, t.delivery_node, false });
            for (int i = pos_d; i < n; ++i) candidate.push_back(a.sequence[i]);

            const float new_cost = sequence_bpr_time(a.current_node, candidate, step);
            if (!std::isfinite(new_cost)) continue;
            const float delta = new_cost - base_cost;
            if (delta < best.cost_delta) {
                best.cost_delta = delta;
                best.pos_p = pos_p;
                best.pos_d = pos_d;
                best.resulting_sequence = candidate;
                if (delta < running_best) running_best = delta;
            }
        }
    }

    return best;
}

// ── Begin (or replan) the leg toward the next stop in the sequence ──────────
// If the sequence is empty, the agent is IDLE. Otherwise, BPR-A* a fresh path
// from current_node to sequence[0].node at `step`. This re-evaluates the path
// under current congestion at every call — exactly the FaithfulCA spirit.
bool FaithfulCACapacitedSolver::begin_leg_bpr(AgentState& a, int step) {
    if (a.sequence.empty()) {
        a.current_path_nodes.clear();
        a.current_path_edges.clear();
        a.next_idx = 0;
        return true;
    }
    const osmium::object_id_type target = a.sequence.front().node;

    // Trivial path edge case (already at next stop): fire arrival immediately.
    if (a.current_node == target) {
        on_arrival(a, step);
        return true;
    }

    BPRPath p = bpr_lookup(a.current_node, target, step);
    if (!p.valid || p.nodes.size() < 2) return false;

    a.current_path_nodes = p.nodes;
    a.current_path_edges = p.edges;
    a.next_idx = 0;
    a.current_edge_t_enter = step;
    a.arrival_step_next_node = edge_arrival_step(a.current_path_edges.front(), step);
    // Congestion footprint registered for the FULL remaining sequence by
    // recommit_route() (Option O commit_plan-style), not edge-by-edge here.
    return true;
}

// ── Arrival event: pop sequence[0], fire pickup or delivery, then recurse ───
// Mirrors FaithfulCASolver's pickup/delivery firing logic, generalised to a
// sequence-popping loop. After firing, if the sequence still has stops, the
// next leg is launched immediately (which itself may collapse to another
// instant on_arrival if the next stop is current_node — e.g. pickup+delivery
// at the same intersection).
void FaithfulCACapacitedSolver::on_arrival(AgentState& a, int step) {
    if (a.sequence.empty()) return;
    const SequenceStop s = a.sequence.front();
    a.sequence.erase(a.sequence.begin());

    if (s.task_id < 0 || s.task_id >= static_cast<int>(tasks_.size())) {
        // Shouldn't happen — defensive no-op.
        return;
    }
    TaskRecord& t = tasks_[s.task_id];

    if (s.is_pickup) {
        t.picked_step = step;
        a.in_flight_task_ids.push_back(s.task_id);
        if (static_cast<int>(a.in_flight_task_ids.size()) > a.capacity) {
            ++capacity_violations_;
        }
        wait_sum_ += (t.picked_step - t.arrival_step);
    } else {
        if (t.picked_step < 0) ++pairing_violations_;
        t.delivered_step = step;
        latency_sum_ += (t.delivered_step - t.arrival_step);
        trip_sum_    += (t.delivered_step - std::max(t.picked_step, t.arrival_step));
        if (t.pd_road_dist > 0.f) {
            road_pd_sum_ += t.pd_road_dist;
            ++road_pd_count_;
        }
        ++completed_;
        instr_.record_delivery(static_cast<int>(&a - agents_.data()));
        auto it = std::find(a.in_flight_task_ids.begin(),
                            a.in_flight_task_ids.end(),
                            s.task_id);
        if (it != a.in_flight_task_ids.end()) a.in_flight_task_ids.erase(it);
    }

    // Launch the next leg (or settle into IDLE if the sequence is empty).
    begin_leg_bpr(a, step);
}

void FaithfulCACapacitedSolver::advance_agent(AgentState& a, int step) {
    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) return;
    if (step < a.arrival_step_next_node) {
        ++active_steps_sum_;
        return;
    }

    if (ctx_ && a.next_idx < static_cast<int>(a.current_path_edges.size())) {
        const auto& ways = ctx_->geo_box->data.ways;
        auto wit = ways.find(a.current_path_edges[a.next_idx]);
        if (wit != ways.end())
            instr_.record_edge_traversal(wit->second.distance_meters);
    }

    ++a.next_idx;
    a.current_node = a.current_path_nodes[a.next_idx];

    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) {
        // Reached the end of the current path = arrived at sequence[0].node.
        on_arrival(a, step);
        recommit_route(a, step);   // plan changed → refresh full-route footprint
        return;
    }

    a.current_edge_t_enter = step;
    const osmium::object_id_type next_edge = a.current_path_edges[a.next_idx];
    a.arrival_step_next_node = edge_arrival_step(next_edge, step);
    ++active_steps_sum_;
}

// Re-register the agent's full remaining sequence on the shared CongestionMap,
// exactly like Option O's commit_plan: unregister the previous footprint, then
// add every edge of every remaining leg with free-flow windows weighted by
// load_per_agent. Called on every plan change.
void FaithfulCACapacitedSolver::recommit_route(AgentState& a, int step) {
    if (!ctx_ || !ctx_->congestion_map || !ctx_->geo_box) return;
    std::vector<osmium::object_id_type> stops;
    stops.reserve(a.sequence.size());
    for (const auto& s : a.sequence) stops.push_back(s.node);
    commit_agent_route(
        *ctx_->congestion_map, *ctx_->geo_box, ctx_->speed_mps,
        a.current_node, stops, step, a.committed_occ,
        [this, step](osmium::object_id_type f, osmium::object_id_type to, int /*t*/) {
            return bpr_lookup(f, to, step).edges;
        });
}

// ── Allocation (paper §3.4 γ-weighting + §3.3 β_π tie-break) ────────────────
// Same selection pipeline as FaithfulCASolver, with two extensions:
//   (1) eligibility includes agents with in_flight < capacity (no busy ban);
//   (2) cost = γ × ΔBPR-sequence-time from cheapest_insertion (instead of
//       single-leg γ × BPR(loc → pickup → delivery)).
// γ-weighting and β_π tie-break logic are preserved verbatim.
bool FaithfulCACapacitedSolver::try_allocate_one(int step) {
    if (pending_task_ids_.empty()) return false;

    std::vector<int> eligible_idx;
    eligible_idx.reserve(agents_.size());
    for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
        const AgentState& a = agents_[i];
        if (static_cast<int>(a.in_flight_task_ids.size()) >= a.capacity) continue;
        eligible_idx.push_back(i);
    }
    if (eligible_idx.empty()) return false;

    struct Cand {
        int   agent_id;
        int   task_idx;
        float cost;            // γ-weighted insertion delta
        int   load;            // in-flight count (β_π tie-break)
        InsertionResult ins;
    };
    std::vector<Cand> cands;
    cands.reserve(eligible_idx.size() * pending_task_ids_.size());

    // ── Perf: top-K agent prefilter + early-termination cutoff ──────────────
    // For each pending task we (1) restrict eligible agents to the K closest
    // by EUCLIDEAN distance to the task's pickup (admissible lower bound on
    // BPR travel time), and (2) feed each cheapest_insertion call a running
    // best raw-delta cutoff so that pos_p iterations whose lower bound
    // already exceeds the cutoff are skipped. Both prunings are near-zero
    // fidelity loss: the BPR-optimal agent is virtually always among the top-K
    // closest by euclidean (BPR factor ≥ 1 means euclidean / speed lower-
    // bounds BPR time), and the early termination only skips strictly
    // dominated candidates.
    float best_cost_so_far = std::numeric_limits<float>::max();
    const float speed = ctx_ ? std::max(0.1f, ctx_->speed_mps) : 1.f;

    // Helper: euclidean lower bound on BPR-trip-time from `from` to `to`.
    auto euclid_time = [&](osmium::object_id_type from,
                            osmium::object_id_type to) -> float {
        if (!ctx_ || !ctx_->pathfinder || from == 0 || to == 0)
            return std::numeric_limits<float>::max();
        const float h_m = const_cast<Pathfinder*>(ctx_->pathfinder)
                              ->heuristic(from, to);
        return h_m / speed;
    };

    for (int ti = 0; ti < static_cast<int>(pending_task_ids_.size()); ++ti) {
        const int tid = pending_task_ids_[ti];
        const TaskRecord& t = tasks_[tid];

        // Build (agent_id, euclidean_time_to_pickup) for all eligible agents,
        // then keep the K smallest. K=0 disables the filter.
        struct AgentEuclid { int ai; float lb; };
        std::vector<AgentEuclid> ranked;
        ranked.reserve(eligible_idx.size());
        for (int ai : eligible_idx) {
            const AgentState& a = agents_[ai];
            ranked.push_back({ ai, euclid_time(a.current_node, t.pickup_node) });
        }
        const int K = hparams.max_candidate_agents;
        if (K > 0 && static_cast<int>(ranked.size()) > K) {
            std::partial_sort(ranked.begin(), ranked.begin() + K, ranked.end(),
                [](const AgentEuclid& x, const AgentEuclid& y){
                    return x.lb < y.lb;
                });
            ranked.resize(static_cast<size_t>(K));
        }

        for (const auto& r : ranked) {
            const AgentState& a = agents_[r.ai];
            const bool is_busy = !a.in_flight_task_ids.empty();
            const float gamma = is_busy ? hparams.gamma_busy : hparams.gamma_idle;

            // Translate the cross-(agent, task) γ-weighted cutoff into a
            // RAW-delta cutoff for this agent's γ. Anything above this raw
            // delta cannot beat best_cost_so_far when γ-weighted.
            const float cutoff_raw = (best_cost_so_far < std::numeric_limits<float>::max())
                ? best_cost_so_far / std::max(0.001f, gamma)
                : std::numeric_limits<float>::max();

            InsertionResult ins = cheapest_insertion(a, t, step, cutoff_raw);
            if (!std::isfinite(ins.cost_delta) || ins.pos_p < 0) continue;

            const float c = gamma * ins.cost_delta;
            if (c < best_cost_so_far) best_cost_so_far = c;
            cands.push_back({ r.ai, ti, c,
                              static_cast<int>(a.in_flight_task_ids.size()),
                              std::move(ins) });
        }
    }
    if (cands.empty()) return false;

    // Find global minimum cost.
    float best_cost = std::numeric_limits<float>::max();
    for (const auto& c : cands) if (c.cost < best_cost) best_cost = c.cost;
    // β_π band — same formula as FaithfulCASolver. For negative or zero
    // best_cost (insertion at end of empty sequence has delta = full trip
    // time, so always positive in practice), the band collapses sensibly.
    const float band = (best_cost > 0.f) ? best_cost * hparams.beta_tie_band
                                          : best_cost + 1e-3f;

    int   best_aid       = -1;
    int   best_tidx      = -1;
    int   best_load      = std::numeric_limits<int>::max();
    float best_band_cost = std::numeric_limits<float>::max();
    int   best_cand_idx  = -1;
    for (int ci = 0; ci < static_cast<int>(cands.size()); ++ci) {
        const auto& c = cands[ci];
        if (c.cost > band) continue;
        bool better = false;
        if (c.load < best_load) better = true;
        else if (c.load == best_load && c.cost < best_band_cost) better = true;
        else if (c.load == best_load && c.cost == best_band_cost &&
                 (best_aid < 0 ||
                  std::make_pair(c.agent_id, c.task_idx) <
                  std::make_pair(best_aid, best_tidx))) better = true;
        if (better) {
            best_load      = c.load;
            best_band_cost = c.cost;
            best_aid       = c.agent_id;
            best_tidx      = c.task_idx;
            best_cand_idx  = ci;
        }
    }
    if (best_aid < 0 || best_tidx < 0 || best_cand_idx < 0) return false;

    // Commit insertion: replace the agent's sequence with the winning one,
    // remove the task from pending, and replan the current leg toward the
    // (possibly new) sequence[0].
    const int tid = pending_task_ids_[best_tidx];
    pending_task_ids_.erase(pending_task_ids_.begin() + best_tidx);
    AgentState& a = agents_[best_aid];
    TaskRecord& t = tasks_[tid];
    t.assigned_agent = best_aid;
    a.sequence = std::move(cands[best_cand_idx].ins.resulting_sequence);

    // The new sequence[0] may differ from the agent's current target. Restart
    // the leg from current_node under fresh BPR-A* at `step`.
    a.current_path_nodes.clear();
    a.current_path_edges.clear();
    a.next_idx = 0;
    const bool ok = begin_leg_bpr(a, step);
    if (!ok) {
        // Insertion path planning failed — roll back: drop the just-inserted
        // task from the sequence and re-pend it. Other tasks already in the
        // sequence stay (they were already committed prior to this attempt).
        auto it = std::find_if(a.sequence.begin(), a.sequence.end(),
            [tid](const SequenceStop& s){ return s.task_id == tid; });
        while (it != a.sequence.end()) {
            it = a.sequence.erase(it);
            it = std::find_if(it, a.sequence.end(),
                [tid](const SequenceStop& s){ return s.task_id == tid; });
        }
        t.assigned_agent = -1;
        pending_task_ids_.push_back(tid);
        // Try to replan the (still valid) prior sequence.
        begin_leg_bpr(a, step);
        recommit_route(a, step);
        return false;
    }
    recommit_route(a, step);   // register the full remaining-sequence footprint
    return true;
}

void FaithfulCACapacitedSolver::step(int timestep) {
    // Flush BPR-A* memoisation at the boundary of every step — the
    // CongestionMap may have advanced since the last allocation, so cached
    // BPR distances from previous steps are stale.
    if (cache_valid_at_step_ != timestep) {
        bpr_cache_.clear();
        cache_valid_at_step_ = timestep;
    }

    if (ctx_) {
        instr_.sample_congestion(ctx_->congestion_map);
        if (ctx_->ghost) instr_.sample_ghost(ctx_->ghost->n_active_now());
    }

    while (try_allocate_one(timestep)) { /* drain pending */ }

    // Robustness: an agent may hold a non-empty sequence but NO active leg if a
    // prior begin_leg_bpr() failed (transient pathfinding miss, or rollback in
    // try_allocate_one). Without this retry it would stay permanently STUCK
    // with undelivered tasks (advance loop below skips empty-path agents).
    // Retried at most once per step; if current_node == sequence[0].node this
    // also fires the deferred arrival event. Idle agents (empty sequence) and
    // mid-leg agents (non-empty path) are untouched.
    for (auto& a : agents_) {
        if (!a.sequence.empty() && a.current_path_edges.empty()) {
            begin_leg_bpr(a, timestep);
            recommit_route(a, timestep);
        }
    }

    for (auto& a : agents_) {
        if (a.current_path_edges.empty()) continue;
        if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) continue;
        advance_agent(a, timestep);
    }
}

SolverMetrics FaithfulCACapacitedSolver::finalize() {
    SolverMetrics m;
    m.tasks_appeared  = appeared_;
    m.tasks_completed = completed_;
    m.tasks_refused   = refused_;
    m.throughput_rate = (appeared_ > 0)
        ? std::min(1.f, static_cast<float>(completed_) / appeared_)
        : 0.f;
    m.accept_rate     = (appeared_ > 0)
        ? static_cast<float>(appeared_ - refused_) / appeared_
        : 0.f;
    if (completed_ > 0) {
        m.latency_mean    = static_cast<double>(latency_sum_) / completed_;
        m.mean_wait_steps = static_cast<double>(wait_sum_)    / completed_;
        m.mean_trip_steps = static_cast<double>(trip_sum_)    / completed_;
    }
    if (road_pd_count_ > 0) {
        m.mean_road_pd_m = road_pd_sum_ / road_pd_count_;
    }
    if (ctx_ && ctx_->n_active_agents > 0 && ctx_->total_steps > 0) {
        m.agent_utilisation = static_cast<float>(active_steps_sum_) /
            (static_cast<float>(ctx_->n_active_agents) * ctx_->total_steps);
    }
    m.capacity_violations = capacity_violations_;
    m.pairing_violations  = pairing_violations_;
    if (ctx_ && ctx_->n_active_agents > 0)
        m.latency_per_agent = m.latency_mean / ctx_->n_active_agents;
    instr_.finalize_into(m);
    return m;
}
