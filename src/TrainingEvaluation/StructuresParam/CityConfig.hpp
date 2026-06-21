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
    std::string osm_file;    // source .pbf filename (may cover multiple cities)
    std::string osm_path;    // full path: <osm_root>/<osm_file>  (set by CityRegistry)
    GeoBBox     bbox;
    CityRole    role;
    double      area_km2;    // approximate, used for density normalisation
    float       ref_lambda;  // reference task/step rate at medium phase

    bool is_train() const { return role == CityRole::TrainAndApply; }
};

// ── City registry (7 cities) ──────────────────────────────────────────────────
//
// Call set_osm_root() once before first use.
// Each city points to a regional .pbf file (see osm_file); the bbox filter is
// applied at parse time by libosmium — no pre-extraction (osmium-tool) needed.
//
// Required files in <osm_root>/:
//   kanto-latest.osm.pbf   — Tokyo + Kyoto   (geofabrik: asia/japan/kanto)
//   kyushu-latest.osm.pbf  — Fukuoka         (geofabrik: asia/japan/kyushu)
//   california-latest.osm.pbf — LosAngeles   (geofabrik: north-america/us/california)
//   new-york-latest.osm.pbf   — NewYork      (geofabrik: north-america/us/new-york)
//   ile-de-france-latest.osm.pbf — Paris     (geofabrik: europe/france/ile-de-france)
//   great-britain-latest.osm.pbf — London    (geofabrik: europe/great-britain)
//
// All 7 cities train and eval: Tokyo | Kyoto | LosAngeles | NewYork | Paris | London | Fukuoka
struct CityRegistry {
    static const std::vector<CityConfig>& all();

    static std::vector<const CityConfig*> train_cities();
    static std::vector<const CityConfig*> comparison_cities();

    static void               set_osm_root(const std::string& root);
    static const std::string& osm_root();
};

#endif // TRAINING_CITY_CONFIG_HPP
