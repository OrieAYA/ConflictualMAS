#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "Box.hpp"
#include <string>

// Fonction de test original (pas de paramètres)
int test();

// Exemples d'utilisation avec paramètres configurables
void create_save_render(const std::string& osm_file,
                       double min_lon, double min_lat, double max_lon, double max_lat,
                       const std::string& cache_path,
                       const std::string& output_name = "example1_map",
                       int width = 2000, int height = 2000);

void load_and_render(const std::string& cache_path,
                    const std::string& output_name_small = "example2_small",
                    const std::string& output_name_large = "example2_large",
                    int width_small = 1000, int height_small = 1000,
                    int width_large = 4000, int height_large = 4000);

void with_flickr_objectives(const std::string& osm_file,
                           double min_lon, double min_lat, double max_lon, double max_lat,
                           const std::string& cache_path,
                           const FlickrConfig& flickr_config,
                           const std::string& output_name = "flickr_map",
                           int width = 2000, int height = 2000);

void complete_workflow(const std::string& osm_file,
                      double min_lon, double min_lat, double max_lon, double max_lat,
                      const std::string& cache_dir,
                      const std::string& cache_prefix = "workflow",
                      const std::string& output_name = "workflow_result",
                      int width = 2000, int height = 2000);

#endif // UTILITY_HPP