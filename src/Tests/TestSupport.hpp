#ifndef TESTS_TEST_SUPPORT_HPP
#define TESTS_TEST_SUPPORT_HPP

#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include <cmath>
#include <iostream>
#include <string>

// Shared helpers for the test batteries (A–D). Header-only, kept deliberately
// small so each battery reads top-to-bottom without indirection.

// Fail-fast check: on the first violated condition, prints "[FAIL] msg" and
// returns false from the enclosing `bool run_*` battery.
#define CHECK(cond, msg) \
    do { if (!(cond)) { std::cout << "  [FAIL] " msg "\n"; return false; } } while (0)

inline bool in01(float v)  { return std::isfinite(v) && v >= 0.f && v <= 1.f; }
inline bool isfin(float v) { return std::isfinite(v); }

// Load (or build + cache) the shared ~4×5 km central-Tokyo "smoke" GeoBox used
// by every battery — same bbox as the training smoke test, so the single cache
// file is reused across all tests.
inline GeoBox load_smoke_geobox(const std::string& osm_file,
                                const std::string& cache_dir) {
    const std::string cache_path = cache_dir + "/smoke_test.json";
    GeoBox gb;
    if (GeoBoxManager::cache_exists(cache_path)) {
        gb = GeoBoxManager::load_geobox(cache_path);
    } else {
        gb = create_geo_box(osm_file, 139.695, 35.670, 139.740, 35.710);
        if (gb.is_valid) GeoBoxManager::save_geobox(gb, cache_path);
    }
    return gb;
}

#endif // TESTS_TEST_SUPPORT_HPP
