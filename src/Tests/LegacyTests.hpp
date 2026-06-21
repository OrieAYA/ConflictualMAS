#ifndef LEGACY_TESTS_HPP
#define LEGACY_TESTS_HPP

#include <string>

void legacy_run_global    (const std::string& cache_dir);
void legacy_run_vns       (const std::string& cache_dir);
void legacy_run_pso       (const std::string& cache_dir);
void legacy_create_geobox (const std::string& osm_file, const std::string& cache_dir);
void legacy_init_poi      (const std::string& cache_dir);
void legacy_render        (const std::string& cache_dir);

#endif // LEGACY_TESTS_HPP
