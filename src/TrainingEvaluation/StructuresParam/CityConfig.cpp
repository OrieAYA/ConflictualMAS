#include "TrainingEvaluation/StructuresParam/CityConfig.hpp"
static std::string s_root = "C:/osm";

void               CityRegistry::set_osm_root(const std::string& r) { s_root = r; }
const std::string& CityRegistry::osm_root()                          { return s_root; }

const std::vector<CityConfig>& CityRegistry::all() {
    // Rebuilt lazily; osm_path uses s_root at first-access time.
    static std::vector<CityConfig> reg;
    if (!reg.empty()) return reg;

    // osm_file: regional .pbf downloaded from geofabrik — no pre-extraction needed.
    // libosmium applies the bbox filter at parse time; result is cached as .json.
    auto add = [&](const char* name, const char* osm_file, GeoBBox bb,
                   CityRole role, double area, float lam) {
        reg.push_back({ name, osm_file,
                        s_root + "/" + osm_file,
                        bb, role, area, lam });
    };

    // ── Training: 3 city families × 2 scales (Small/Medium) ────────────────
    // Paper protocol trains on Tokyo, Los Angeles, Paris (Small + Medium each).
    // Tokyo  : Shibuya/Shinjuku core (Small) & Yamanote inner (Medium)
    // LA     : Downtown core (Small) & Mid-Wilshire/Hollywood (Medium)
    // Paris  : Île-de-la-Cité core (Small) & inner arrondissements (Medium)
    // Kyoto  : kept for held-out generalisation (Small + Medium)
    // Size classes share a common centre; area scales with SCE so the graph
    // size (hence congestion density count/hot_ways) tracks SCE:
    //   Small = 1× (~25 km²) | Medium = 2× (~50 km²) | Large = 5× (~125 km²).
    // Medium/Large bboxes = Small scaled by √2 / √5 about the centre.
    add("Tokyo_Small",       "Tokyo.osm.pbf",      { 139.680, 35.650, 139.740, 35.700 }, CityRole::TrainAndApply,   25.0, 0.06f);
    add("Tokyo_Medium",      "Tokyo.osm.pbf",      { 139.668, 35.640, 139.752, 35.710 }, CityRole::TrainAndApply,   50.0, 0.06f);
    add("LosAngeles_Small",  "LosAngeles.osm.pbf", {-118.280, 34.030,-118.220, 34.070 }, CityRole::TrainAndApply,   25.0, 0.05f);
    add("LosAngeles_Medium", "LosAngeles.osm.pbf", {-118.292, 34.022,-118.208, 34.078 }, CityRole::TrainAndApply,   50.0, 0.05f);
    add("Paris_Small",       "Paris.osm.pbf",      {   2.320, 48.830,   2.380, 48.880 }, CityRole::TrainAndApply,   25.0, 0.08f);
    add("Paris_Medium",      "Paris.osm.pbf",      {   2.308, 48.820,   2.392, 48.890 }, CityRole::TrainAndApply,   50.0, 0.08f);
    // Kyoto kept as comparison/generalisation (was a training family before).
    add("Kyoto_Small",       "Kyoto.osm.pbf",      { 135.740, 34.990, 135.790, 35.030 }, CityRole::ComparisonOnly,  25.0, 0.04f);
    add("Kyoto_Medium",      "Kyoto.osm.pbf",      { 135.730, 34.982, 135.800, 35.038 }, CityRole::ComparisonOnly,  50.0, 0.04f);

    // ── Comparison / generalisation only ────────────────────────────────────
    // Large variants (SCE 3, ~75 km²) + the NewYork family — same centres as
    // their Smalls, half-extents ×√3 (Large) / ×√2 (Medium).
    add("Tokyo_Large",       "Tokyo.osm.pbf",      { 139.658, 35.632, 139.762, 35.718 }, CityRole::ComparisonOnly,  75.0, 0.06f);
    add("Kyoto_Large",       "Kyoto.osm.pbf",      { 135.722, 34.975, 135.808, 35.045 }, CityRole::ComparisonOnly,  75.0, 0.04f);
    add("LosAngeles_Large",  "LosAngeles.osm.pbf", {-118.302, 34.015,-118.198, 34.085 }, CityRole::ComparisonOnly,  75.0, 0.05f);
    add("Paris_Large",       "Paris.osm.pbf",      {   2.298, 48.812,   2.402, 48.898 }, CityRole::ComparisonOnly,  75.0, 0.08f);
    add("NewYork_Medium",    "NewYork.osm.pbf",    { -74.022, 40.700, -73.938, 40.770 }, CityRole::ComparisonOnly,  50.0, 0.07f);
    add("NewYork_Large",     "NewYork.osm.pbf",    { -74.032, 40.692, -73.928, 40.778 }, CityRole::ComparisonOnly,  75.0, 0.07f);
    add("Fukuoka",     "Fukuoka.osm.pbf",      { 130.30, 33.55, 130.55, 33.70 }, CityRole::ComparisonOnly,  340.0, 0.04f);
    add("NewYork",     "NewYork.osm.pbf",      { -74.05, 40.60, -73.75, 40.85 }, CityRole::ComparisonOnly,  783.0, 0.07f);
    add("Paris",       "Paris.osm.pbf",        {   2.25, 48.80,   2.45, 48.92 }, CityRole::ComparisonOnly,  105.0, 0.08f);
    add("London",      "London.osm.pbf",       {  -0.25, 51.45,   0.00, 51.60 }, CityRole::ComparisonOnly, 1572.0, 0.05f);

    // ── Generalisation Small variant (~25 km², same scale as train Small) ──
    // Cropped 5×5 km bbox around Manhattan (Midtown → Wall Street) so the
    // per-episode compute matches the train Smalls (~2 min/ep on TP). Held out
    // from training for the cross-city generalisation eval. (Paris_Small is now
    // a training city — see the training block above.)
    add("NewYork_Small", "NewYork.osm.pbf",    { -74.01, 40.71, -73.95, 40.76 }, CityRole::ComparisonOnly,   25.0, 0.07f);

    return reg;
}

std::vector<const CityConfig*> CityRegistry::train_cities() {
    std::vector<const CityConfig*> out;
    for (const auto& c : all()) if ( c.is_train()) out.push_back(&c);
    return out;
}

std::vector<const CityConfig*> CityRegistry::comparison_cities() {
    std::vector<const CityConfig*> out;
    for (const auto& c : all()) if (!c.is_train()) out.push_back(&c);
    return out;
}
