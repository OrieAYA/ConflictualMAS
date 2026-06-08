#include "OperableEnvironment.hpp"
#include "DMASforPD/GlobalMemory/GlobalMemory.hpp"
#include <limits>

// ---- Task management ---------------------------------------------------

void OperableEnvironment::add_task(const PDPTask& task) {
    int old_n = static_cast<int>(nodes.size());
    nodes.push_back(task.pickup);
    nodes.push_back(task.delivery);
    int new_n = static_cast<int>(nodes.size());

    index_map_[task.pickup.id]   = old_n;
    index_map_[task.delivery.id] = old_n + 1;

    std::vector<float> new_costs(new_n * new_n, -1.0f);
    for (int i = 0; i < old_n; ++i)
        for (int j = 0; j < old_n; ++j)
            new_costs[i * new_n + j] = costs_[i * old_n + j];
    costs_ = std::move(new_costs);
}

void OperableEnvironment::add_single_node(const ObjectiveNode& node) {
    if (index_map_.count(node.id)) return;  // already present
    int old_n = static_cast<int>(nodes.size());
    nodes.push_back(node);
    int new_n = old_n + 1;
    index_map_[node.id] = old_n;

    std::vector<float> new_costs(new_n * new_n, -1.0f);
    for (int i = 0; i < old_n; ++i)
        for (int j = 0; j < old_n; ++j)
            new_costs[i * new_n + j] = costs_[i * old_n + j];
    costs_ = std::move(new_costs);
}

void OperableEnvironment::remove_task(osmium::object_id_type pickup_id,
                                       osmium::object_id_type delivery_id) {
    int old_n = static_cast<int>(nodes.size());

    std::vector<int> survivors;
    survivors.reserve(old_n);
    for (int k = 0; k < old_n; ++k)
        if (nodes[k].id != pickup_id && nodes[k].id != delivery_id)
            survivors.push_back(k);

    int new_n = static_cast<int>(survivors.size());
    std::vector<float> new_costs(new_n * new_n, -1.0f);
    for (int ni = 0; ni < new_n; ++ni)
        for (int nj = 0; nj < new_n; ++nj)
            new_costs[ni * new_n + nj] = costs_[survivors[ni] * old_n + survivors[nj]];

    std::vector<ObjectiveNode> new_nodes;
    new_nodes.reserve(new_n);
    for (int k : survivors) new_nodes.push_back(nodes[k]);

    nodes  = std::move(new_nodes);
    costs_ = std::move(new_costs);

    // Rebuild index map from the new nodes vector.
    index_map_.clear();
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
        index_map_[nodes[i].id] = i;
}

// ---- Lookup ------------------------------------------------------------

int OperableEnvironment::find_index(osmium::object_id_type node_id) const {
    auto it = index_map_.find(node_id);
    return it != index_map_.end() ? it->second : -1;
}

int OperableEnvironment::size() const {
    return static_cast<int>(nodes.size());
}

// ---- Cost access -------------------------------------------------------

void OperableEnvironment::set_cost(int i, int j, float cost) {
    costs_[i * static_cast<int>(nodes.size()) + j] = cost;
}

float OperableEnvironment::get_cost(int i, int j) const {
    return costs_[i * static_cast<int>(nodes.size()) + j];
}

bool OperableEnvironment::has_cost(int i, int j) const {
    return get_cost(i, j) >= 0.0f;
}

// ---- Cost refresh ------------------------------------------------------

void OperableEnvironment::refresh_costs(PDPGlobalMemory& memory) {
    int n = static_cast<int>(nodes.size());
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j || has_cost(i, j)) continue;
            const ObjectivePath* path = memory.get_or_compute_path(
                nodes[i].id, nodes[j].id, nodes[i].group_id);
            if (path && path->valid())
                set_cost(i, j, path->cost);
        }
    }
}

// ---- Time-aware (congestion-BPR) cost ----------------------------------

void OperableEnvironment::set_time_context(
    PDPGlobalMemory& memory, float speed_mps, int base_step
) {
    mem_ctx_   = &memory;
    speed_ctx_ = (speed_mps > 0.f) ? speed_mps : 1.f;
    base_ctx_  = base_step;
    tcost_memo_.clear();
}

float OperableEnvironment::get_cost_at(int i, int j, int depart_step) const {
    const float static_c = get_cost(i, j);
    if (!mem_ctx_ || static_c < 0.f) return static_c;   // no context / infeasible

    // Memoise per (i, j, departure step). kBucket = 1 → EXACT per-step
    // resolution (the load profile is indexed by integer step, so 1 is the
    // finest meaningful granularity; 0 would be undefined). This re-costs every
    // branch at its exact traversal time — no coarse collapsing. Cost stays
    // bounded because re-exploring the SAME decomposition node yields the SAME
    // exact depart_step → the memo still hits; only genuinely different-time
    // evaluations of the same (i,j) edge are recomputed, which is the intended
    // precision. The memo is cleared each replan (set_time_context), so
    // forecasts are always recomputed fresh per planning call.
    constexpr int kBucket = 1;                           // steps per bucket (1 = exact)
    const int n      = static_cast<int>(nodes.size());
    const int bucket = (depart_step > 0 ? depart_step : 0) / kBucket;
    const long long key =
        (static_cast<long long>(i * n + j) << 21) ^ static_cast<long long>(bucket);
    auto it = tcost_memo_.find(key);
    if (it != tcost_memo_.end()) return it->second;

    const float c = mem_ctx_->bpr_path_cost(
        nodes[i].id, nodes[j].id, nodes[i].group_id, bucket * kBucket, speed_ctx_);
    const float result = (c < std::numeric_limits<float>::max()) ? c : static_c;
    tcost_memo_[key] = result;
    return result;
}
