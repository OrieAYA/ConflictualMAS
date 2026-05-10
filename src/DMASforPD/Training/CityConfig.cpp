#include "CityConfig.hpp"
#include <stdexcept>

static std::string s_osm_root = "C:/osm";

void CityRegistry::set_osm_root(const std::string& root) { s_osm_root = root; }
const std::string& CityRegistry::osm_root()              { return s_osm_root; }

const std::vector<CityConfig>& CityRegistry::all() {
    static std::vector<CityConfig> registry = [] {
        std::vector<CityConfig> v;

        // ── Train + Apply ──────────────────────────────────────────────────

        v.push_back({
            "Tokyo",
            s_osm_root + "/tokyo.osm.pbf",
            { 139.60, 35.55, 139.90, 35.80 },
            CityRole::TrainAndApply,
            /*area_km2=*/ 627.0,
            /*ref_lambda=*/ 0.06f
        });

        v.push_back({
            "Kyoto",
            s_osm_root + "/kyoto.osm.pbf",
            { 135.65, 34.95, 135.85, 35.10 },
            CityRole::TrainAndApply,
            /*area_km2=*/ 217.0,
            /*ref_lambda=*/ 0.04f
        });

        v.push_back({
            "LosAngeles",
            s_osm_root + "/los_angeles.osm.pbf",
            { -118.50, 33.90, -118.10, 34.15 },
            CityRole::TrainAndApply,
            /*area_km2=*/ 1300.0,
            /*ref_lambda=*/ 0.05f
        });

        // ── Comparison only ────────────────────────────────────────────────

        v.push_back({
            "NewYork",
            s_osm_root + "/new_york.osm.pbf",
            { -74.05, 40.60, -73.75, 40.85 },
            CityRole::ComparisonOnly,
            /*area_km2=*/ 783.0,
            /*ref_lambda=*/ 0.07f
        });

        v.push_back({
            "Paris",
            s_osm_root + "/paris.osm.pbf",
            { 2.25, 48.80, 2.45, 48.92 },
            CityRole::ComparisonOnly,
            /*area_km2=*/ 105.0,
            /*ref_lambda=*/ 0.08f
        });

        v.push_back({
            "London",
            s_osm_root + "/london.osm.pbf",
            { -0.25, 51.45, 0.00, 51.60 },
            CityRole::ComparisonOnly,
            /*area_km2=*/ 1572.0,
            /*ref_lambda=*/ 0.05f
        });

        v.push_back({
            "Fukuoka",
            s_osm_root + "/fukuoka.osm.pbf",
            { 130.30, 33.55, 130.55, 33.70 },
            CityRole::ComparisonOnly,
            /*area_km2=*/ 340.0,
            /*ref_lambda=*/ 0.04f
        });

        return v;
    }();
    return registry;
}

std::vector<const CityConfig*> CityRegistry::train_cities() {
    std::vector<const CityConfig*> out;
    for (const auto& c : all())
        if (c.is_train()) out.push_back(&c);
    return out;
}

std::vector<const CityConfig*> CityRegistry::comparison_cities() {
    std::vector<const CityConfig*> out;
    for (const auto& c : all())
        if (!c.is_train()) out.push_back(&c);
    return out;
}
