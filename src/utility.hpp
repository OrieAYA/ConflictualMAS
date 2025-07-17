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

void with_flickr_objectives(const std::string& osm_file,
                           double min_lon, double min_lat, double max_lon, double max_lat,
                           const std::string& cache_path,
                           const FlickrConfig& flickr_config,
                           const std::string& output_name = "flickr_map",
                           int width = 2000, int height = 2000);

void complete_workflow(const std::string& osm_file,
                      double min_lon, double min_lat, double max_lon, double max_lat,
                      const std::string& cache_dir,
                      const std::string& cache_prefix,
                      const std::string& output_name,
                      int width, int height,
                      const FlickrConfig& flickr_config,
                      bool use_flickr_objectives = true);

#endif // UTILITY_HPP