#include "CityConfig.hpp"

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

    // ── Training: 3 Tokyo scales (curriculum-friendly: short → long tasks) ──
    // Small  ~  5 ×  5 km — Shibuya/Shinjuku core, dense, short trips.
    // Medium ~ 12 × 12 km — Yamanote inner wards, mixed density.
    // Large  ~ 30 × 28 km — current bbox, full metropolitan span.
    add("Tokyo_Small",  "Tokyo.osm.pbf", { 139.68, 35.65, 139.74, 35.70 }, CityRole::TrainAndApply,   25.0, 0.06f);
    add("Tokyo_Medium", "Tokyo.osm.pbf", { 139.64, 35.62, 139.80, 35.74 }, CityRole::TrainAndApply,  144.0, 0.06f);
    add("Tokyo_Large",  "Tokyo.osm.pbf", { 139.60, 35.55, 139.90, 35.80 }, CityRole::TrainAndApply,  627.0, 0.06f);

    // ── Comparison / generalisation only ────────────────────────────────────
    add("Kyoto",      "Kyoto.osm.pbf",      { 135.65, 34.95, 135.85, 35.10 }, CityRole::ComparisonOnly,  217.0, 0.04f);
    add("Fukuoka",    "Fukuoka.osm.pbf",    { 130.30, 33.55, 130.55, 33.70 }, CityRole::ComparisonOnly,  340.0, 0.04f);
    add("LosAngeles", "LosAngeles.osm.pbf", {-118.50, 33.90,-118.10, 34.15 }, CityRole::ComparisonOnly, 1300.0, 0.05f);
    add("NewYork",    "NewYork.osm.pbf",    { -74.05, 40.60, -73.75, 40.85 }, CityRole::ComparisonOnly,  783.0, 0.07f);
    add("Paris",      "Paris.osm.pbf",      {   2.25, 48.80,   2.45, 48.92 }, CityRole::ComparisonOnly,  105.0, 0.08f);
    add("London",     "London.osm.pbf",     {  -0.25, 51.45,   0.00, 51.60 }, CityRole::ComparisonOnly, 1572.0, 0.05f);

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
