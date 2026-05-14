#ifndef MULTI_CITY_TRAINER_HPP
#define MULTI_CITY_TRAINER_HPP

#include "TrainingConfig.hpp"
#include "TrainingLogger.hpp"
#include "EpisodeRunner.hpp"
#include "CityConfig.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <memory>
#include <string>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// CityAssets — heap-stable resources for one city
// ════════════════════════════════════════════════════════════════════════════
//
// GeoBox and Pathfinder both own/reference large graph data.
// Pathfinder holds GeoBox by reference, so CityAssets must never be
// moved or copied after construction — allocate as unique_ptr<CityAssets>.
//
// ep_cfg is stored here so EpisodeRunner (which holds const EpisodeConfig&)
// always has a valid referent for the lifetime of the asset.
struct CityAssets {
    const CityConfig* config = nullptr;
    int               index  = 0;
    EpisodeConfig     ep_cfg;       // stable referent for EpisodeRunner
    GeoBox            geo_box;
    Pathfinder        pathfinder;   // holds GeoBox& geo_box above

    CityAssets(const CityConfig* cc, int idx, EpisodeConfig ep, GeoBox&& gb)
        : config(cc), index(idx), ep_cfg(std::move(ep)),
          geo_box(std::move(gb)), pathfinder(geo_box) {}

    CityAssets(const CityAssets&)            = delete;
    CityAssets& operator=(const CityAssets&) = delete;
    CityAssets(CityAssets&&)                 = delete;
    CityAssets& operator=(CityAssets&&)      = delete;
};

// ════════════════════════════════════════════════════════════════════════════
// MultiCityTrainer
// ════════════════════════════════════════════════════════════════════════════
//
// Full experimental protocol:
//   1. Load (or cache) all 7 city graphs exactly once.
//   2. For each independent seed:
//        a. Re-initialise MAPPO policy weights (Xavier, seeded).
//        b. Optionally load a pre-trained checkpoint.
//        c. Round-robin training over the 3 train cities (n_rounds rounds).
//        d. Periodic + final evaluation on all 7 cities × 3 policy modes.
//        e. Log every episode to per-seed CSV + cross-seed summary.csv.
//        f. Save the trained policy checkpoint.
//
// One EpisodeRunner is kept alive per city throughout a seed so that the
// A* path cache (group-0 in PDPGlobalMemory) warms up across episodes.
// City graphs are shared across seeds — only the runner (and hence the
// path cache) is recreated per seed.
class MultiCityTrainer {
public:
    void train(const TrainingConfig& cfg);

private:
    static std::unique_ptr<CityAssets> load_city(const CityConfig& cc, int idx,
                                                  EpisodeConfig ep,
                                                  const std::string& cache_root);

    static int run_eval(const TrainingConfig& cfg,
                        const std::vector<std::unique_ptr<CityAssets>>& assets,
                        std::vector<std::unique_ptr<EpisodeRunner>>& runners,
                        int global_ep, int seed,
                        TrainingLogger& logger);

    static int run_generalize_eval(const TrainingConfig& cfg,
                                   const std::vector<std::unique_ptr<CityAssets>>& gen_assets,
                                   std::vector<std::unique_ptr<EpisodeRunner>>& gen_runners,
                                   int global_ep, int seed,
                                   TrainingLogger& logger);
};

#endif // MULTI_CITY_TRAINER_HPP
