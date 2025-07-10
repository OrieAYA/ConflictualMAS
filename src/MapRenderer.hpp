#ifndef MAP_RENDERER_HPP
#define MAP_RENDERER_HPP

#include "Box.hpp"
#include <osmium/osm/box.hpp>
#include <string>

// Fonction indépendante pour rendre une map depuis une GeoBox
bool render_map(const GeoBox& geo_box,
                const std::string& output_filename,
                int width = 2000,
                int height = 2000);

// Fonction indépendante pour rendre une map depuis MyData directement
bool render_map_from_data(const MyData& data,
                         const osmium::Box& bbox,
                         const std::string& output_filename,
                         int width = 2000,
                         int height = 2000);

// Classe MapRenderer (optionnelle, pour compatibilité)
class MapRenderer {
public:
    MapRenderer() = default;
    
    void render_shibuya_map(const MyData& data_collector,
                           const osmium::Box& map_bbox,
                           const std::string& output_filename,
                           int width = 2000,
                           int height = 2000) {
        render_map_from_data(data_collector, map_bbox, output_filename, width, height);
    }
};

#endif // MAP_RENDERER_HPP