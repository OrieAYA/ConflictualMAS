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
    // Tokyo  : Shibuya/Shinjuku core (Small) & Yamanote inner (Medium)
    // Kyoto  : Karasuma core (Small) & Kyoto basin (Medium)
    // LA     : Downtown core (Small) & Mid-Wilshire/Hollywood (Medium)
    // Each "Small" ≈ 5×5 km / 25 km² and each "Medium" ≈ 12×12 km / 140 km².
    // Tokyo_Large is kept as comparison-only because per-episode time on the
    // 1M-node graph dominates the schedule (~3-15 min/episode).
    add("Tokyo_Small",       "Tokyo.osm.pbf",      { 139.68, 35.65, 139.74, 35.70 }, CityRole::TrainAndApply,   25.0, 0.06f);
    add("Tokyo_Medium",      "Tokyo.osm.pbf",      { 139.64, 35.62, 139.80, 35.74 }, CityRole::TrainAndApply,  144.0, 0.06f);
    add("Kyoto_Small",       "Kyoto.osm.pbf",      { 135.74, 34.99, 135.79, 35.03 }, CityRole::TrainAndApply,   25.0, 0.04f);
    add("Kyoto_Medium",      "Kyoto.osm.pbf",      { 135.70, 34.95, 135.83, 35.07 }, CityRole::TrainAndApply,  140.0, 0.04f);
    add("LosAngeles_Small",  "LosAngeles.osm.pbf", {-118.28, 34.03,-118.22, 34.07 }, CityRole::TrainAndApply,   25.0, 0.05f);
    add("LosAngeles_Medium", "LosAngeles.osm.pbf", {-118.32, 34.00,-118.18, 34.12 }, CityRole::TrainAndApply,  140.0, 0.05f);

    // ── Comparison / generalisation only ────────────────────────────────────
    add("Tokyo_Large", "Tokyo.osm.pbf",        { 139.60, 35.55, 139.90, 35.80 }, CityRole::ComparisonOnly,  627.0, 0.06f);
    add("Fukuoka",     "Fukuoka.osm.pbf",      { 130.30, 33.55, 130.55, 33.70 }, CityRole::ComparisonOnly,  340.0, 0.04f);
    add("NewYork",     "NewYork.osm.pbf",      { -74.05, 40.60, -73.75, 40.85 }, CityRole::ComparisonOnly,  783.0, 0.07f);
    add("Paris",       "Paris.osm.pbf",        {   2.25, 48.80,   2.45, 48.92 }, CityRole::ComparisonOnly,  105.0, 0.08f);
    add("London",      "London.osm.pbf",       {  -0.25, 51.45,   0.00, 51.60 }, CityRole::ComparisonOnly, 1572.0, 0.05f);

    // ── Generalisation Small variants (~25 km², same scale as train Small) ──
    // Cropped 5×5 km bbox around each city's core so the per-episode compute
    // matches Tokyo_Small/Kyoto_Small/LosAngeles_Small (~2 min/ep on TP).
    // Used by Option O cross-method comparison to avoid the ~30-100× slowdown
    // that full Paris/NewYork OSM graphs incur on TP's Dijkstra-per-task.
    // Paris_Small   covers central Paris (Notre-Dame / 1er-4e arrondissements)
    // NewYork_Small covers mid/lower Manhattan (Midtown → Wall Street)
    add("Paris_Small",   "Paris.osm.pbf",      {   2.32, 48.83,   2.38, 48.88 }, CityRole::ComparisonOnly,   25.0, 0.08f);
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
