#ifndef SOTA_SOLVER_CONGESTION_HPP
#define SOTA_SOLVER_CONGESTION_HPP

#include "Environment/Congestion/CongestionMap.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// Solver congestion footprint — replicates the DMAS pipeline's commit_plan
// ════════════════════════════════════════════════════════════════════════════
//
// In Option O (EpisodeRunner + PDPGlobalMemory::commit_plan) every agent
// publishes the occupancy of its ENTIRE remaining route on the shared
// CongestionMap: each edge of each planned leg is registered for its
// FREE-FLOW window [t_enter, t_exit] with weight = load_per_agent, and the
// previous registration is removed on every re-plan. This dense, predictive,
// load_per_agent-weighted footprint is what produces a meaningful per-edge BPR.
//
// The standalone SoTA solvers previously registered only the CURRENT edge as
// they moved (sparse, weight 1) → load almost never exceeded capacity → BPR ≈ 1.
// This helper lets every solver reproduce the Option O footprint EXACTLY so the
// throughput / latency / BPR axes are measured under a homogeneous congestion
// regime across O and OP/OQ.

// One committed edge occupancy: (edge_id, t_enter, t_exit). Stored per agent so
// the previous registration can be removed before re-committing.
using CommittedOcc = std::vector<std::tuple<osmium::object_id_type, int, int>>;

// Register an agent's full remaining route on `cmap`, exactly like commit_plan:
//   1. remove the agent's previous occupancy (`committed`);
//   2. walk the ordered `stops` from `from`, leg by leg, accumulating FREE-FLOW
//      time (ceil(edge_length / speed)); add_agent each edge for [t, t+steps]
//      with weight = load_per_agent; remember it in `committed`.
// `path_fn(from, to, t)` returns the edge-id list of the planned leg path.
template <typename PathFn>
inline void commit_agent_route(
    CongestionMap& cmap, const GeoBox& geo_box, float speed_mps,
    osmium::object_id_type from,
    const std::vector<osmium::object_id_type>& stops,
    int start_step, CommittedOcc& committed, PathFn&& path_fn)
{
    const int w = std::max(1, cmap.params.load_per_agent);

    // 1. Unregister the previous footprint.
    for (const auto& occ : committed)
        cmap.remove_agent(std::get<0>(occ), std::get<1>(occ), std::get<2>(occ), w);
    committed.clear();

    // 2. Register the full remaining route with free-flow windows.
    const auto& ways = geo_box.data.ways;
    const float sp   = std::max(0.1f, speed_mps);
    osmium::object_id_type cur = from;
    int t = start_step;
    for (osmium::object_id_type target : stops) {
        if (target == 0)      continue;
        if (target == cur)    { cur = target; continue; }
        const auto edges = path_fn(cur, target, t);
        for (osmium::object_id_type eid : edges) {
            auto it = ways.find(eid);
            if (it == ways.end()) continue;
            const int steps = std::max(1, static_cast<int>(
                std::ceil(it->second.distance_meters / sp)));
            cmap.add_agent(eid, t, t + steps, w);
            committed.emplace_back(eid, t, t + steps);
            t += steps;
        }
        cur = target;
    }
}

#endif // SOTA_SOLVER_CONGESTION_HPP
