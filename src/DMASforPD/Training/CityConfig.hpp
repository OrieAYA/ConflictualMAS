#ifndef CITY_CONFIG_HPP
#define CITY_CONFIG_HPP

#include <string>
#include <vector>

// ── City role in the experimental protocol ────────────────────────────────────
enum class CityRole {
    TrainAndApply,   // Tokyo, Kyoto, Los Angeles — policy trained here, also evaluated
    ComparisonOnly   // New York, Paris, London, Fukuoka — evaluation / generalisation only
};

// ── Geographic bounding box (WGS-84) ─────────────────────────────────────────
struct GeoBBox {
    double min_lon, min_lat, max_lon, max_lat;
};

// ── Single city descriptor ────────────────────────────────────────────────────
struct CityConfig {
    std::string name;        // display name
    std::string osm_path;    // path to .osm.pbf file (user-configurable)
    GeoBBox     bbox;        // road network bounding box
    CityRole    role;

    // Approximate area km² (used for density normalisation)
    double area_km2;

    // Typical task density reference (tasks/km²/hour at medium phase)
    float ref_lambda;

    bool is_train() const { return role == CityRole::TrainAndApply; }
};

// ── City registry ─────────────────────────────────────────────────────────────
//
// OSM file paths follow the pattern  <osm_root>/<name>.osm.pbf.
// Call CityRegistry::set_osm_root() once before use.
//
// Train cities (indices 0-2) : Tokyo, Kyoto, Los Angeles
// Comparison cities (3-6)    : New York, Paris, London, Fukuoka
struct CityRegistry {
    static const std::vector<CityConfig>& all();

    // Subset helpers
    static std::vector<const CityConfig*> train_cities();
    static std::vector<const CityConfig*> comparison_cities();

    // Set the root directory that contains the .osm.pbf files.
    // Default: "C:/osm"
    static void set_osm_root(const std::string& root);
    static const std::string& osm_root();
};

#endif // CITY_CONFIG_HPP
