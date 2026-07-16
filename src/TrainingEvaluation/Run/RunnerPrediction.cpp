#include "TrainingEvaluation/Run/Runner.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <algorithm>
#include <cmath>

// ── LSM prediction / signaling ────────────────────────────────────────────────

namespace {

inline float bpr_norm(float mul) {
    const float t = std::max(0.f, mul - 1.f);
    return t / (1.f + t);
}

// 32×32 grid cell → 8×8 coarse cell.
inline int coarse_of(int cell) {
    return ((cell >> 5) >> 2) * kLsmCoarseDim + ((cell & 31) >> 2);
}

}  // namespace

void EpisodeRunner::lsm_tick(int step, int total_steps, float lambda,
                             int /*n_active*/) {
    LsmModule& lsm = lsm_module();
    const RegionStatsGrid& grid = memory_.region_grid;

    float u[kLsmIn] = {};
    {
        float cong[kLsmCells] = {}, dens[kLsmCells] = {};
        int   cnt[kLsmCells]  = {};
        const float inv_max = 1.f / std::max(1, grid.max_count);
        for (int c = 0; c < RegionStatsGrid::kSize; ++c) {
            const int cc = coarse_of(c);
            cong[cc] += bpr_norm(grid.cell_cong_cache[c]);
            dens[cc] += std::min(1.f, grid.task_counts[c] * inv_max);
            cnt[cc]  += 1;
        }
        for (int cc = 0; cc < kLsmCells; ++cc) {
            const float inv = 1.f / std::max(1, cnt[cc]);
            u[cc]             = cong[cc] * inv;
            u[kLsmCells + cc] = dens[cc] * inv;
        }
    }
    const auto ls = memory_.congestion_map.load_sample_now();
    int n_act_agents = 0;
    for (const auto& a : all_agents_)
        if (a && a->status == AgentStatus::Active) ++n_act_agents;
    float* g = u + 2 * kLsmCells;
    g[0] = (total_steps > 0)
        ? static_cast<float>(step) / total_steps : 0.f;
    g[1] = std::clamp(ls.mean / 5.f, 0.f, 1.f);
    g[2] = std::clamp(ls.peak / 10.f, 0.f, 1.f);
    g[3] = std::clamp(memory_.congestion_map.n_edges_load_ge(2) / 500.f,
                      0.f, 1.f);
    g[4] = std::clamp(ghost_traffic_.n_active_now() / 500.f, 0.f, 1.f);
    g[5] = std::clamp(lambda, 0.f, 1.f);
    g[6] = (episode_fleet_size_ > 0)
        ? std::min(1.f, static_cast<float>(n_act_agents) / episode_fleet_size_)
        : 0.f;
    g[7] = (memory_.count_total() > 0)
        ? static_cast<float>(memory_.count_available()) / memory_.count_total()
        : 0.f;

    lsm.step(u);
    ++lsm_stats_.n_ticks;

    // Delayed supervision: the realized coarse congestion NOW is the target
    // for the liquid states recorded H steps ago.
    lsm_hist_.emplace_back(step, lsm.state());
    while (!lsm_hist_.empty()
           && lsm_hist_.front().first + cfg_.lsm_horizon <= step) {
        if (cfg_.lsm_train) {
            float y[kLsmOut];
            for (int cc = 0; cc < kLsmCells; ++cc) y[cc] = u[cc];
            y[kLsmCells] = g[1];
            lsm_stats_.mse_sum +=
                lsm.learn(lsm_hist_.front().second, y, cfg_.lsm_lr);
            ++lsm_stats_.n_learn;
        }
        lsm_hist_.pop_front();
    }

    float yhat[kLsmOut];
    lsm.predict(yhat);
    float alert[kLsmCells];
    int n_alert = 0;
    for (int cc = 0; cc < kLsmCells; ++cc) {
        const float v = std::clamp(yhat[cc], 0.f, 1.f);
        alert[cc] = (v >= cfg_.lsm_alert_threshold) ? v : 0.f;
        if (alert[cc] > 0.f) ++n_alert;
    }
    lsm_stats_.alert_cells_sum += n_alert;

    lsm_alert_.clear();
    if (n_alert == 0) return;

    const auto& nodes = memory_.geo_box.data.nodes;
    auto route_alert = [&](const std::vector<osmium::object_id_type>& seq,
                           int from, float& best) {
        const int n = static_cast<int>(seq.size());
        if (from >= n) return;
        const int stride = std::max(1, (n - from) / 16);
        for (int i = from; i < n; i += stride) {
            auto it = nodes.find(seq[i]);
            if (it == nodes.end()) continue;
            const int cc = coarse_of(
                grid.cell_of(it->second.lat, it->second.lon));
            best = std::max(best, alert[cc]);
        }
    };
    for (const auto& a : all_agents_) {
        if (!a || a->status != AgentStatus::Active) continue;
        float best = 0.f;
        if (a->edge_cursor)
            route_alert(a->edge_cursor->nodes, a->edge_cursor->next_idx, best);
        const ObjectivePath* np = a->local_memory.next_path;
        if (np && np->valid()) route_alert(np->nodes, 0, best);
        if (best > 0.f) lsm_alert_[a->agent_id] = best;
    }
    lsm_stats_.agent_alerts_sum += static_cast<long long>(lsm_alert_.size());
}
