#include "CityConfig.hpp"

static std::string s_root = "C:/osm";

void               CityRegistry::set_osm_root(const std::string& r) { s_root = r; }
const std::string& CityRegistry::osm_root()                          { return s_root; }

const std::vector<CityConfig>& CityRegistry::all() {
    // Rebuilt lazily; osm_path uses s_root at first-access time.
    static std::vector<CityConfig> reg;
    if (!reg.empty()) return reg;

    auto add = [&](const char* name, GeoBBox bb, CityRole role,
                   double area, float lam) {
        reg.push_back({ name,
                        s_root + "/" + name + ".osm.pbf",
                        bb, role, area, lam });
    };

    // ── Train + Apply ──────────────────────────────────────────────────────
    add("Tokyo",       { 139.60, 35.55, 139.90, 35.80 }, CityRole::TrainAndApply,  627.0,  0.06f);
    add("Kyoto",       { 135.65, 34.95, 135.85, 35.10 }, CityRole::TrainAndApply,  217.0,  0.04f);
    add("LosAngeles",  {-118.50, 33.90,-118.10, 34.15 }, CityRole::TrainAndApply, 1300.0,  0.05f);

    // ── Comparison only ────────────────────────────────────────────────────
    add("NewYork",     { -74.05, 40.60, -73.75, 40.85 }, CityRole::ComparisonOnly, 783.0,  0.07f);
    add("Paris",       {   2.25, 48.80,   2.45, 48.92 }, CityRole::ComparisonOnly, 105.0,  0.08f);
    add("London",      {  -0.25, 51.45,   0.00, 51.60 }, CityRole::ComparisonOnly,1572.0,  0.05f);
    add("Fukuoka",     { 130.30, 33.55, 130.55, 33.70 }, CityRole::ComparisonOnly, 340.0,  0.04f);

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
