#include "SoTA/Standalone/HAPC.hpp"
#include "Environment/Simulation/CongestionMap.hpp"
#include "Environment/GeoBox/GraphSearch.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr float kInf = std::numeric_limits<float>::infinity();
}



int HybridAdaptivePredictiveSolver::ZonePairForecast::cell_of(
    osmium::object_id_type id, const GeoBox& g) const
{
    if (lat_step <= 0.f || lon_step <= 0.f) return -1;
    auto it = g.data.nodes.find(id);
    if (it == g.data.nodes.end()) return -1;
    const float lat = static_cast<float>(it->second.lat);
    const float lon = static_cast<float>(it->second.lon);
    int cy = static_cast<int>((lat - lat_min) / lat_step);
    int cx = static_cast<int>((lon - lon_min) / lon_step);
    cy = std::clamp(cy, 0, dim - 1);
    cx = std::clamp(cx, 0, dim - 1);
    return cy * dim + cx;
}

void HybridAdaptivePredictiveSolver::ZonePairForecast::register_arrival(
    int p_cell, int d_cell)
{
    if (p_cell < 0 || d_cell < 0) return;
    const int64_t key = static_cast<int64_t>(p_cell) * (dim * dim) + d_cell;
    count_interval[key] += 1.f;
    count_total[key]    += 1.f;
    ++n_interval;
}

void HybridAdaptivePredictiveSolver::ZonePairForecast::advance_interval(int step)
{
    const int idx = step / std::max(1, interval_len);
    if (idx == interval_index) return;
    interval_index = idx;
    count_interval.clear();
    n_interval = 0;
}

std::vector<HybridAdaptivePredictiveSolver::ZonePairForecast::VirtualCall>
HybridAdaptivePredictiveSolver::ZonePairForecast::top_h(
    int H, float min_p, int min_samples) const
{
    if (H <= 0) return {};
    // thin interval -> fall back to cumulative counts
    const auto& src = (n_interval >= min_samples) ? count_interval : count_total;
    if (src.empty()) return {};

    // Collect candidates (filter cells with no rep_node — unrealised on graph).
    std::vector<std::pair<float, std::pair<int,int>>> cands;
    cands.reserve(src.size());
    const int n2 = dim * dim;
    for (const auto& [key, v] : src) {
        if (v <= 0.f) continue;
        const int p_cell = static_cast<int>(key / n2);
        const int d_cell = static_cast<int>(key % n2);
        if (p_cell >= static_cast<int>(rep_node_of_cell.size()) ||
            d_cell >= static_cast<int>(rep_node_of_cell.size())) continue;
        if (rep_node_of_cell[p_cell] == 0 || rep_node_of_cell[d_cell] == 0)
            continue;
        cands.emplace_back(v, std::make_pair(p_cell, d_cell));
    }
    if (cands.empty()) return {};

    // top-H by mass; ties on cell indices for determinism
    auto by_mass = [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    };
    if (static_cast<int>(cands.size()) > H) {
        std::partial_sort(cands.begin(), cands.begin() + H, cands.end(), by_mass);
        cands.resize(static_cast<size_t>(H));
    } else {
        std::sort(cands.begin(), cands.end(), by_mass);
    }

    float sum = 0.f;
    for (const auto& c : cands) sum += c.first;
    if (sum <= 0.f) return {};

    // prune then normalise: eq.15 needs sum p_h = 1 exactly
    std::vector<std::pair<float, std::pair<int,int>>> kept;
    kept.reserve(cands.size());
    float kept_sum = 0.f;
    for (const auto& c : cands) {
        if (c.first / sum < min_p) continue;
        kept.push_back(c);
        kept_sum += c.first;
    }
    if (kept.empty() || kept_sum <= 0.f) { kept = cands; kept_sum = sum; }

    std::vector<VirtualCall> out;
    out.reserve(kept.size());
    for (const auto& c : kept) {
        VirtualCall vc;
        vc.pickup_node   = rep_node_of_cell[c.second.first];
        vc.delivery_node = rep_node_of_cell[c.second.second];
        vc.probability   = c.first / kept_sum;
        out.push_back(vc);
    }
    return out;
}



HybridAdaptivePredictiveSolver::SharedDistanceCache&
HybridAdaptivePredictiveSolver::shared_cache()
{
    // Reusable storage, wiped at every init() (one episode's worth of
    static SharedDistanceCache instance;
    return instance;
}

std::unique_ptr<PathHelper>&
HybridAdaptivePredictiveSolver::shared_path_helper()
{
    // PathHelper rebuilt at every init() so its A* result cache holds only the
    static std::unique_ptr<PathHelper> instance;
    return instance;
}

void HybridAdaptivePredictiveSolver::init(const SolverContext& ctx) {
    ctx_   = &ctx;

    // cold caches every episode: bounded, and timings stay fair
    {
        auto& sc  = shared_cache();
        auto& sph = shared_path_helper();
        sc.gb = ctx.geo_box;
        sc.gb_node_count = (ctx.geo_box) ? ctx.geo_box->data.nodes.size() : 0;
        sc.dijkstra_from.clear();
        sc.entries = 0;
        sc.prewarm_refused = 0;
        sph = std::make_unique<PathHelper>(*ctx.geo_box);
        paths_ = sph.get();  // non-owning view for this instance
    }

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

    tasks_.clear();
    appeared_ = completed_ = refused_ = 0;
    latency_sum_ = wait_sum_ = trip_sum_ = 0;
    road_pd_sum_ = 0.0;
    road_pd_count_ = 0;
    active_steps_sum_ = 0;
    wait_count_ = 0;
    capacity_violations_ = pairing_violations_ = 0;
    instr_.init(ctx.n_active_agents, ctx.speed_mps);

    //  Build the OD-pair forecast grid over the GeoBox bounding box.
    forecast_.dim = std::max(2, hparams.demand_grid_dim);
    forecast_.count_interval.clear();
    forecast_.count_total.clear();
    forecast_.n_interval     = 0;
    forecast_.interval_index = -1;
    // Paper §4.1 splits the horizon into equal time intervals ΔT (three hourly
    forecast_.interval_len = std::max(
        1, ctx.total_steps / std::max(1, hparams.forecast_n_intervals));
    const int n_cells = forecast_.dim * forecast_.dim;
    forecast_.rep_node_of_cell.assign(static_cast<size_t>(n_cells), 0);
    if (ctx.geo_box && !ctx.geo_box->data.nodes.empty()) {
        float lat_lo =  1e9f, lat_hi = -1e9f;
        float lon_lo =  1e9f, lon_hi = -1e9f;
        for (const auto& [_, node] : ctx.geo_box->data.nodes) {
            const float lat = static_cast<float>(node.lat);
            const float lon = static_cast<float>(node.lon);
            if (lat < lat_lo) lat_lo = lat;
            if (lat > lat_hi) lat_hi = lat;
            if (lon < lon_lo) lon_lo = lon;
            if (lon > lon_hi) lon_hi = lon;
        }
        forecast_.lat_min  = lat_lo;
        forecast_.lon_min  = lon_lo;
        const float pad = 1e-6f;
        forecast_.lat_step = std::max(pad, (lat_hi - lat_lo)) / forecast_.dim;
        forecast_.lon_step = std::max(pad, (lon_hi - lon_lo)) / forecast_.dim;

        // Assign representatives — first incident-on-a-way node found per cell.
        for (const auto& [nid, node] : ctx.geo_box->data.nodes) {
            if (node.incident_ways.empty()) continue;  // skip isolated nodes
            const int c = forecast_.cell_of(nid, *ctx.geo_box);
            if (c < 0 || c >= n_cells) continue;
            if (forecast_.rep_node_of_cell[c] == 0)
                forecast_.rep_node_of_cell[c] = nid;
        }
    }
}

void HybridAdaptivePredictiveSolver::inject_task(const ScheduledTask& task,
                                                  int step) {
    TaskRecord r;
    r.task_id        = static_cast<int>(tasks_.size());
    r.pickup_node    = task.pickup_node_id;
    r.delivery_node  = task.delivery_node_id;
    r.arrival_step   = step;

    //  Bulk cache pre-warm (Dijkstra-from-source)
    prewarm_from_node(r.pickup_node);
    prewarm_from_node(r.delivery_node);

    r.pd_road_dist = seg_cost(r.pickup_node, r.delivery_node);
    if (!std::isfinite(r.pd_road_dist)) r.pd_road_dist = 0.f;

    tasks_.push_back(r);
    ++appeared_;

    register_arrival(r.pickup_node, r.delivery_node);

    // Paper §3.1 — dispatch fires immediately on call arrival (event-driven).
    instr_.time_allocation([&]{
        try_insert_task(r.task_id, step);
    });
}



void HybridAdaptivePredictiveSolver::prewarm_from_node(
    osmium::object_id_type src) const
{
    if (src == 0) return;
    auto& sc = shared_cache();
    if (sc.dijkstra_from.count(src)) return;  // already done this episode
    if (!ctx_ || !ctx_->geo_box) return;

    // budget guard: refuse rather than grow; seg_cost falls back to A*
    const size_t projected = sc.entries + ctx_->geo_box->data.nodes.size();
    if (projected * kPrewarmBytesPerEntry > kPrewarmBudgetBytes) {
        ++sc.prewarm_refused;
        return;
    }

    // Single-source Dijkstra over the road graph. Result is d(src, X) for
    auto dists = graph_search::dijkstra_distances(*ctx_->geo_box, src);
    auto& dist_map = sc.dijkstra_from[src];
    dist_map.reserve(dists.size());
    for (const auto& [node, dist] : dists) {
        dist_map.emplace(node, dist);
    }
    sc.entries += dist_map.size();
}



float HybridAdaptivePredictiveSolver::seg_cost(
    osmium::object_id_type from, osmium::object_id_type to) const
{
    if (from == 0 || to == 0) return kInf;
    if (from == to) return 0.f;

    //  L1: per-source Dijkstra cache (hot path, ~150ns)
    auto& sc = shared_cache();
    auto it = sc.dijkstra_from.find(from);
    if (it != sc.dijkstra_from.end()) {
        auto jt = it->second.find(to);
        if (jt != it->second.end()) return jt->second;
    }
    // undirected: try the reverse
    it = sc.dijkstra_from.find(to);
    if (it != sc.dijkstra_from.end()) {
        auto jt = it->second.find(from);
        if (jt != it->second.end()) return jt->second;
    }

    //  L2 fallback: PathHelper (persistent across episodes within city)
    if (!paths_) return kInf;
    const auto& p = paths_->get(from, to);
    return p.valid ? p.cost : kInf;
}

HybridAdaptivePredictiveSolver::PaperCost
HybridAdaptivePredictiveSolver::paper_cost(
    osmium::object_id_type from,
    const std::vector<SequenceStop>& seq,
    int initial_load,
    float alpha) const
{
    PaperCost cb;
    if (seq.empty()) return cb;

    osmium::object_id_type cur = from;
    int load = initial_load;  // L_j^(i-1) at i = 1: load BEFORE first stop
    float cum_time = 0.f;     // T_j^i - T_j^0 — relative to current position

    for (const auto& stop : seq) {
        const float seg = seg_cost(cur, stop.node);
        if (!std::isfinite(seg)) {
            cb.travel  = kInf;
            cb.waiting = kInf;
            return cb;
        }

        // eq.11 travel: (load+1) * segment time
        cb.travel += static_cast<float>(load + 1) * seg;

        cum_time += seg;

        // eq.11 waiting: fires at pickups only
        if (stop.is_pickup) {
            cb.waiting += alpha * cum_time;
        }

        load += stop.is_pickup ? 1 : -1;
        cur = stop.node;
    }
    return cb;
}



bool HybridAdaptivePredictiveSolver::build_profile(
    const std::vector<SequenceStop>& seq,
    int initial_load,
    osmium::object_id_type from_node,
    SeqProfile& out) const
{
    const int n = static_cast<int>(seq.size());
    out.valid = false;
    out.from_node = from_node;
    out.initial_load = initial_load;
    out.t_seg.assign(static_cast<size_t>(n), 0.f);
    out.load_before.assign(static_cast<size_t>(n), 0);
    out.load_after.assign(static_cast<size_t>(n), 0);
    out.arrive_t.assign(static_cast<size_t>(n), 0.f);
    out.t_seg_prefix.assign(static_cast<size_t>(n + 1), 0.f);
    out.pickup_suffix.assign(static_cast<size_t>(n + 1), 0);

    osmium::object_id_type prev = from_node;
    int   load = initial_load;
    float cum  = 0.f;
    for (int i = 0; i < n; ++i) {
        const float s = seg_cost(prev, seq[i].node);
        if (!std::isfinite(s)) return false;
        out.t_seg[i]       = s;
        out.load_before[i] = load;
        load              += seq[i].is_pickup ? 1 : -1;
        out.load_after[i]  = load;
        cum               += s;
        out.arrive_t[i]    = cum;
        prev               = seq[i].node;
    }
    for (int i = 0; i < n; ++i)
        out.t_seg_prefix[i + 1] = out.t_seg_prefix[i] + out.t_seg[i];
    for (int i = n - 1; i >= 0; --i)
        out.pickup_suffix[i] = out.pickup_suffix[i + 1] + (seq[i].is_pickup ? 1 : 0);

    out.valid = true;
    return true;
}



HybridAdaptivePredictiveSolver::InsertionResult
HybridAdaptivePredictiveSolver::cheapest_insertion_in_seq(
    const std::vector<SequenceStop>& sequence,
    int initial_load,
    int capacity,
    osmium::object_id_type from_node,
    osmium::object_id_type virtual_pickup,
    osmium::object_id_type virtual_delivery) const
{
    // Single-shot variant — builds the profile, then dispatches. For the
    SeqProfile profile;
    if (!build_profile(sequence, initial_load, from_node, profile)) {
        InsertionResult bad;
        bad.cost_delta = kInf;
        return bad;
    }
    return cheapest_insertion_with_profile(
        sequence, profile, capacity, virtual_pickup, virtual_delivery);
}

HybridAdaptivePredictiveSolver::InsertionResult
HybridAdaptivePredictiveSolver::cheapest_insertion_with_profile(
    const std::vector<SequenceStop>& sequence,
    const SeqProfile& profile,
    int capacity,
    osmium::object_id_type virtual_pickup,
    osmium::object_id_type virtual_delivery) const
{
    InsertionResult best;
    best.cost_delta = kInf;
    best.pos_p = best.pos_d = -1;

    if (!profile.valid) return best;
    const int n = static_cast<int>(sequence.size());
    const int initial_load = profile.initial_load;
    if (initial_load > capacity) return best;



    const osmium::object_id_type from_node = profile.from_node;
    const auto& t_seg         = profile.t_seg;
    const auto& load_after    = profile.load_after;
    const auto& arrive_t      = profile.arrive_t;
    const auto& t_seg_prefix  = profile.t_seg_prefix;
    const auto& pickup_suffix = profile.pickup_suffix;

    //  Insertion-related seg_cost precompute (O(n))
    auto& seg_to_P  = scratch_seg_to_P_;   seg_to_P.assign (n + 1, kInf);
    auto& seg_P_out = scratch_seg_P_out_;  seg_P_out.assign(std::max(1, n), kInf);
    auto& seg_D_in  = scratch_seg_D_in_;   seg_D_in.assign (n + 1, kInf);
    auto& seg_D_out = scratch_seg_D_out_;  seg_D_out.assign(std::max(1, n), kInf);

    auto pred_node = [&](int k) -> osmium::object_id_type {
        return (k > 0) ? sequence[k - 1].node : from_node;
    };
    for (int k = 0; k <= n; ++k) seg_to_P[k] = seg_cost(pred_node(k), virtual_pickup);
    for (int k = 0; k <  n; ++k) seg_P_out[k] = seg_cost(virtual_pickup, sequence[k].node);
    for (int k = 1; k <= n; ++k) seg_D_in[k]  = seg_cost(sequence[k - 1].node, virtual_delivery);
    for (int k = 0; k <  n; ++k) seg_D_out[k] = seg_cost(virtual_delivery, sequence[k].node);
    const float seg_P_to_D = seg_cost(virtual_pickup, virtual_delivery);
    if (!std::isfinite(seg_P_to_D)) return best;

    const float alpha = hparams.alpha;

    //  Try every (pos_p, pos_d) — O(1) per candidate
    for (int pos_p = 0; pos_p <= n; ++pos_p) {
        const int lbp = (pos_p > 0) ? load_after[pos_p - 1] : initial_load;
        if (lbp + 1 > capacity) continue;

        const float arrive_at_P_pred = (pos_p > 0) ? arrive_t[pos_p - 1] : 0.f;
        const float stp              = seg_to_P[pos_p];
        if (!std::isfinite(stp)) continue;
        const float arrive_at_P      = arrive_at_P_pred + stp;

        for (int pos_d = pos_p; pos_d <= n; ++pos_d) {
            bool feasible = true;
            for (int k = pos_p; k <= pos_d - 1; ++k) {
                if (load_after[k] + 1 > capacity) { feasible = false; break; }
            }
            if (!feasible) continue;

            float delta_travel  = 0.f;
            float delta_waiting = 0.f;

            if (pos_p == pos_d) {
                //  Case A: P and D consecutive at position pos_p
                if (pos_p < n) {
                    const float sdo = seg_D_out[pos_p];
                    if (!std::isfinite(sdo)) continue;
                    delta_travel =
                        - static_cast<float>(lbp + 1) * t_seg[pos_p]
                        + static_cast<float>(lbp + 1) * stp
                        + static_cast<float>(lbp + 2) * seg_P_to_D
                        + static_cast<float>(lbp + 1) * sdo;
                    const float delta_combined = stp + seg_P_to_D + sdo - t_seg[pos_p];
                    delta_waiting =
                        alpha * arrive_at_P
                        + alpha * delta_combined *
                            static_cast<float>(pickup_suffix[pos_p]);
                } else {
                    // append at the end
                    delta_travel =
                        + static_cast<float>(lbp + 1) * stp
                        + static_cast<float>(lbp + 2) * seg_P_to_D;
                    delta_waiting = alpha * arrive_at_P;
                }
            } else {
                //  Case B: pos_p < pos_d
                const float spo = seg_P_out[pos_p];   // P → seq[pos_p]
                const float sdi = seg_D_in[pos_d];    // seq[pos_d-1] → D
                if (!std::isfinite(spo) || !std::isfinite(sdi)) continue;

                const int   lbd = load_after[pos_d - 1];  // load_before[pos_d]
                const float delta_P = stp + spo - t_seg[pos_p];

                // Travel cost — P side
                delta_travel +=
                    - static_cast<float>(lbp + 1) * t_seg[pos_p]
                    + static_cast<float>(lbp + 1) * stp
                    + static_cast<float>(lbp + 2) * spo;

                // middle edges carry P: +1 load each
                delta_travel += t_seg_prefix[pos_d] - t_seg_prefix[pos_p + 1];

                // Travel cost — D side
                float delta_D;
                if (pos_d < n) {
                    const float sdo = seg_D_out[pos_d];
                    if (!std::isfinite(sdo)) continue;
                    delta_travel +=
                        - static_cast<float>(lbd + 1) * t_seg[pos_d]
                        + static_cast<float>(lbd + 2) * sdi
                        + static_cast<float>(lbd + 1) * sdo;
                    delta_D = sdi + sdo - t_seg[pos_d];
                } else {
                    delta_travel +=
                        + static_cast<float>(lbd + 2) * sdi;
                    delta_D = sdi;  // irrelevant for waiting since no pickups in [n, n)
                }

                // Waiting cost — new pickup + shifted originals.
                const int pickups_in_PD     = pickup_suffix[pos_p] - pickup_suffix[pos_d];
                const int pickups_after_D   = pickup_suffix[pos_d];
                delta_waiting =
                    alpha * arrive_at_P
                    + alpha * delta_P * static_cast<float>(pickups_in_PD)
                    + alpha * (delta_P + delta_D) * static_cast<float>(pickups_after_D);
            }

            const float delta = delta_travel + delta_waiting;
            if (delta < best.cost_delta) {
                best.cost_delta = delta;
                best.pos_p = pos_p;
                best.pos_d = pos_d;
            }
        }
    }

    //  Materialise the winning sequence (once, at the end)
    if (best.pos_p >= 0 && best.pos_d >= 0) {
        best.resulting_sequence.clear();
        best.resulting_sequence.reserve(static_cast<size_t>(n + 2));
        for (int i = 0; i < best.pos_p; ++i)
            best.resulting_sequence.push_back(sequence[i]);
        best.resulting_sequence.push_back({-1, virtual_pickup, true});
        for (int i = best.pos_p; i < best.pos_d; ++i)
            best.resulting_sequence.push_back(sequence[i]);
        best.resulting_sequence.push_back({-1, virtual_delivery, false});
        for (int i = best.pos_d; i < n; ++i)
            best.resulting_sequence.push_back(sequence[i]);
    }
    return best;
}

HybridAdaptivePredictiveSolver::InsertionResult
HybridAdaptivePredictiveSolver::cheapest_insertion(
    const AgentState& a, const TaskRecord& t) const
{
    InsertionResult r = cheapest_insertion_in_seq(
        a.sequence,
        static_cast<int>(a.in_flight_task_ids.size()),
        a.capacity,
        a.current_node,
        t.pickup_node,
        t.delivery_node);

    // stamp the task id (helper inserts -1)
    if (r.pos_p >= 0 && r.pos_d >= 0) {
        for (auto& s : r.resulting_sequence) {
            if (s.task_id == -1) s.task_id = t.task_id;
        }
    }
    return r;
}



bool HybridAdaptivePredictiveSolver::try_insert_task(int task_id, int step) {
    if (task_id < 0 || task_id >= static_cast<int>(tasks_.size())) return false;
    TaskRecord& t = tasks_[task_id];
    if (t.assigned_agent >= 0) return true;

    // Pre-compute virtual future calls (paper §4.4 — top-H OD-pairs).
    std::vector<ZonePairForecast::VirtualCall> virtual_calls;
    if (hparams.enable_two_step) {
        virtual_calls = forecast_.top_h(hparams.forecast_top_h,
                                         hparams.forecast_min_p,
                                         hparams.forecast_min_samples);
    }

    int   best_aid  = -1;
    int   best_pp   = -1;
    int   best_pd   = -1;
    float best_total = kInf;
    std::vector<SequenceStop> best_resulting_sequence;

    SeqProfile base_prof;
    SeqProfile post_prof;

    for (int ai = 0; ai < static_cast<int>(agents_.size()); ++ai) {
        const AgentState& a = agents_[ai];
        const int initial_load = static_cast<int>(a.in_flight_task_ids.size());

        if (!build_profile(a.sequence, initial_load, a.current_node, base_prof))
            continue;

        // Immediate insertion (paper C_j(k+1) − C_j(k)).
        InsertionResult now = cheapest_insertion_with_profile(
            a.sequence, base_prof, a.capacity,
            t.pickup_node, t.delivery_node);
        if (!std::isfinite(now.cost_delta)) continue;

        for (auto& s : now.resulting_sequence)
            if (s.task_id == -1) s.task_id = t.task_id;

        // Future expected cost: Σ_h p_h × min_insertion(virtual_h into
        // post-insertion sequence). Paper eq. 15.
        float future = 0.f;
        if (hparams.enable_two_step && !virtual_calls.empty()) {
            const auto& post_seq = now.resulting_sequence;

            if (build_profile(post_seq, initial_load, a.current_node, post_prof)) {
                for (const auto& vc : virtual_calls) {
                    const InsertionResult fut = cheapest_insertion_with_profile(
                        post_seq, post_prof, a.capacity,
                        vc.pickup_node, vc.delivery_node);
                    if (!std::isfinite(fut.cost_delta)) {
                        // Virtual call unservable — penalise heavily so this
                        // outer candidate is discouraged.
                        future += vc.probability * 1e6f;
                        continue;
                    }
                    future += vc.probability * fut.cost_delta;
                }
            }
        }

        const float total = now.cost_delta + future;
        if (total < best_total) {
            best_total = total;
            best_aid   = ai;
            best_pp    = now.pos_p;
            best_pd    = now.pos_d;
            best_resulting_sequence = std::move(now.resulting_sequence);
        }
    }
    if (best_aid < 0) {
        // Failure → exponential backoff so the per-step retry doesn't burn
        const int backoff = 1 << std::min(t.retry_count, 6);
        t.next_retry_step = step + backoff;
        ++t.retry_count;
        return false;
    }

    AgentState& a = agents_[best_aid];
    a.sequence = std::move(best_resulting_sequence);
    t.assigned_agent = best_aid;
    t.next_retry_step = 0;
    t.retry_count     = 0;
    resync_leg(a, step);
    recommit_route(a, step);   // register the full remaining-sequence footprint
    (void)best_pp; (void)best_pd;  // recorded only for potential telemetry
    return true;
}



int HybridAdaptivePredictiveSolver::edge_arrival_step(
    osmium::object_id_type edge_id, int t_enter)
{
    if (!ctx_) return t_enter + 1;
    const auto& ways = ctx_->geo_box->data.ways;
    auto it = ways.find(edge_id);
    if (it == ways.end()) return t_enter + 1;
    const float length_m = it->second.distance_meters;
    const float base_time = length_m / std::max(0.1f, ctx_->speed_mps);
    if (!ctx_->congestion_map)
        return t_enter + std::max(1, static_cast<int>(std::ceil(base_time)));

    // pays the BPR of the n OTHERS (parity with schedule_next_edge)
    const int self_w = std::max(1, ctx_->congestion_map->params.load_per_agent);
    const float adj  = ctx_->congestion_map->adjusted_cost(
        edge_id, base_time, length_m, t_enter, self_w);
    // sampled at ENTRY: advance() purges past steps
    instr_.record_edge_entry(base_time, adj,
                             ctx_->congestion_map->get_load(edge_id, t_enter));
    return t_enter + std::max(1, static_cast<int>(std::ceil(adj)));
}

bool HybridAdaptivePredictiveSolver::resync_leg(AgentState& a, int step) {
    if (a.sequence.empty()) {
        const bool mid_edge =
            !a.current_path_edges.empty() &&
            a.next_idx < static_cast<int>(a.current_path_edges.size()) &&
            a.arrival_step_next_node > step;
        if (!mid_edge) {
            a.current_path_nodes.clear();
            a.current_path_edges.clear();
            a.next_idx = 0;
        }
        return true;
    }

    if (!a.current_path_nodes.empty() && !a.current_path_edges.empty() &&
        a.current_path_nodes.back() == a.sequence.front().node) {
        return true;
    }

    // Mid-edge guard: paper §3.1 says sequences only change at events (call
    // arrival or stop reach). If we're traversing an edge, defer the replan.
    if (!a.current_path_edges.empty() &&
        a.next_idx < static_cast<int>(a.current_path_edges.size()) &&
        a.arrival_step_next_node > step) {
        return true;
    }

    const osmium::object_id_type target = a.sequence.front().node;
    if (a.current_node == target) {
        a.current_path_nodes.clear();
        a.current_path_edges.clear();
        a.next_idx = 0;
        a.arrival_step_next_node = step;
        return true;
    }

    const SimplePath& sp = paths_->get(a.current_node, target);
    if (!sp.valid || sp.nodes.size() < 2) return false;

    a.current_path_nodes = sp.nodes;
    a.current_path_edges = sp.edges;
    a.next_idx = 0;
    a.current_edge_t_enter = step;
    // timed by recommit_route (commit before schedule)
    a.arrival_step_next_node = -1;
    return true;
}

void HybridAdaptivePredictiveSolver::advance_agent(AgentState& a, int step) {
    if (a.sequence.empty()) return;

    if (a.current_path_edges.empty()) {
        if (a.current_node != a.sequence.front().node) {
            resync_leg(a, step);
            recommit_route(a, step);
            return;
        }
        // fall through — fire the stop
    } else {
        if (a.arrival_step_next_node < 0)
            a.arrival_step_next_node =
                edge_arrival_step(a.current_path_edges[a.next_idx], step);
        // route exposure, one sample per in-transit agent per step
        if (ctx_ && ctx_->congestion_map)
            instr_.sample_route_exposure(ctx_->congestion_map->get_load(
                a.current_path_edges[a.next_idx], step));
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

        if (a.next_idx < static_cast<int>(a.current_path_edges.size())) {
            a.current_edge_t_enter = step;
            const osmium::object_id_type next_edge =
                a.current_path_edges[a.next_idx];
            a.arrival_step_next_node = edge_arrival_step(next_edge, step);
            ++active_steps_sum_;
            return;
        }
        a.current_path_nodes.clear();
        a.current_path_edges.clear();
        a.next_idx = 0;
    }

    if (a.current_node != a.sequence.front().node) {
        resync_leg(a, step);
        recommit_route(a, step);
        return;
    }

    const SequenceStop stop = a.sequence.front();
    a.sequence.erase(a.sequence.begin());

    if (stop.task_id < 0 ||
        stop.task_id >= static_cast<int>(tasks_.size())) {
        resync_leg(a, step);
        recommit_route(a, step);
        return;
    }
    TaskRecord& t = tasks_[stop.task_id];
    if (stop.is_pickup) {
        t.picked_step = step;
        a.in_flight_task_ids.push_back(stop.task_id);
        if (static_cast<int>(a.in_flight_task_ids.size()) > a.capacity) {
            ++capacity_violations_;
        }
        wait_sum_ += (t.picked_step - t.arrival_step);
        ++wait_count_;
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
        const int agent_idx = static_cast<int>(&a - agents_.data());
        instr_.record_delivery(agent_idx);
        auto it = std::find(a.in_flight_task_ids.begin(),
                            a.in_flight_task_ids.end(),
                            stop.task_id);
        if (it != a.in_flight_task_ids.end()) a.in_flight_task_ids.erase(it);
    }
    resync_leg(a, step);
    recommit_route(a, step);   // sequence changed (stop reached) → refresh footprint
}

// republish the whole remaining sequence on the shared map
void HybridAdaptivePredictiveSolver::recommit_route(AgentState& a, int step) {
    if (!ctx_ || !ctx_->congestion_map || !ctx_->geo_box || !paths_) return;
    std::vector<osmium::object_id_type> stops;
    stops.reserve(a.sequence.size());
    for (const auto& s : a.sequence) stops.push_back(s.node);

    // first leg: use the running path, a fresh A* could diverge
    bool first_leg = true;
    commit_agent_route(
        *ctx_->congestion_map, *ctx_->geo_box, ctx_->speed_mps,
        a.current_node, stops, step, a.committed_occ,
        [&](osmium::object_id_type f, osmium::object_id_type to, int /*t*/) {
            if (first_leg) {
                first_leg = false;
                if (!a.current_path_edges.empty() &&
                    a.next_idx < static_cast<int>(a.current_path_edges.size()) &&
                    to == a.current_path_nodes.back()) {
                    return std::vector<osmium::object_id_type>(
                        a.current_path_edges.begin() + a.next_idx,
                        a.current_path_edges.end());
                }
            }
            return paths_->get(f, to).edges;
        });

    if (a.arrival_step_next_node < 0 &&
        a.next_idx < static_cast<int>(a.current_path_edges.size()))
        a.arrival_step_next_node =
            edge_arrival_step(a.current_path_edges[a.next_idx], step);
}



void HybridAdaptivePredictiveSolver::register_arrival(
    osmium::object_id_type pickup_node,
    osmium::object_id_type delivery_node)
{
    if (!ctx_) return;
    const int p_cell = forecast_.cell_of(pickup_node,   *ctx_->geo_box);
    const int d_cell = forecast_.cell_of(delivery_node, *ctx_->geo_box);
    forecast_.register_arrival(p_cell, d_cell);
}



void HybridAdaptivePredictiveSolver::step(int timestep) {
    if (ctx_) {
        instr_.sample_congestion(ctx_->congestion_map);
        if (ctx_->ghost) instr_.sample_ghost(ctx_->ghost->n_active_now());
    }

    // Roll the demand forecast over to the ΔT containing this step (paper
    // §4.1: probabilities are constant inside an interval, refreshed between).
    forecast_.advance_interval(timestep);

    // retry tasks whose backoff elapsed
    for (auto& t : tasks_) {
        if (t.assigned_agent < 0 &&
            t.delivered_step < 0 &&
            timestep >= t.next_retry_step) {
            instr_.time_allocation([&]{
                try_insert_task(t.task_id, timestep);
            });
        }
    }

    // no periodic MPC: sequences change on events only
    for (auto& a : agents_) advance_agent(a, timestep);
}



SolverMetrics HybridAdaptivePredictiveSolver::finalize() {
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
        m.mean_trip_steps = static_cast<double>(trip_sum_)    / completed_;
    }
    // Wait is accumulated at every PICKUP → its denominator is the pickup
    // count, not the delivery count (RL parity: Metrics.cpp wait_sum/wait_count).
    if (wait_count_ > 0)
        m.mean_wait_steps = static_cast<double>(wait_sum_) / wait_count_;
    if (road_pd_count_ > 0) {
        m.mean_road_pd_m = road_pd_sum_ / road_pd_count_;
    }
    if (ctx_ && ctx_->n_active_agents > 0 && ctx_->total_steps > 0) {
        m.agent_utilisation = static_cast<float>(active_steps_sum_) /
            (static_cast<float>(ctx_->n_active_agents) * ctx_->total_steps);
    }
    m.capacity_violations = capacity_violations_;
    m.pairing_violations  = pairing_violations_;
    // normalised by mean active agents (RL parity)
    if (ctx_ && ctx_->total_steps > 0) {
        const double mean_active =
            static_cast<double>(active_steps_sum_) / ctx_->total_steps;
        m.latency_per_agent = m.latency_mean / std::max(1.0, mean_active);
    }
    instr_.finalize_into(m);
    return m;
}