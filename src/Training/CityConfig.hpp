#ifndef TRAINING_CITY_CONFIG_HPP
#define TRAINING_CITY_CONFIG_HPP

#include <string>
#include <vector>

// ── City role in the experimental protocol ────────────────────────────────────
enum class CityRole {
    TrainAndApply,   // Tokyo, Kyoto, Los Angeles — MAPPO trained here, also evaluated
    ComparisonOnly   // New York, Paris, London, Fukuoka — generalisation / comparison only
};

struct GeoBBox { double min_lon, min_lat, max_lon, max_lat; };

// ── Single city descriptor ────────────────────────────────────────────────────
struct CityConfig {
    std::string name;
    std::string osm_path;    // set by CityRegistry after set_osm_root()
    GeoBBox     bbox;
    CityRole    role;
    double      area_km2;    // approximate, used for density normalisation
    float       ref_lambda;  // reference task/step rate at medium phase

    bool is_train() const { return role == CityRole::TrainAndApply; }
};

// ── City registry (7 cities) ──────────────────────────────────────────────────
//
// Call set_osm_root() once before first use; OSM files expected at
//   <root>/<name>.osm.pbf   (lower-snake-case, e.g. los_angeles.osm.pbf)
//
// Train (indices 0-2) : Tokyo | Kyoto | LosAngeles
// Compare (indices 3-6): NewYork | Paris | London | Fukuoka
struct CityRegistry {
    static const std::vector<CityConfig>& all();

    static std::vector<const CityConfig*> train_cities();
    static std::vector<const CityConfig*> comparison_cities();

    static void               set_osm_root(const std::string& root);
    static const std::string& osm_root();
};

#endif // TRAINING_CITY_CONFIG_HPP
