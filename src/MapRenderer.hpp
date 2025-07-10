#ifndef MAP_RENDERER_HPP
#define MAP_RENDERER_HPP

#include "Box.hpp"
#include <osmium/osm/box.hpp>
#include <string>

class MapRenderer {
public:
    MapRenderer();

    void render_shibuya_map(
        const MyData& data_collector,
        const osmium::Box& map_bbox,
        const std::string& output_filename,
        int width = 2000,
        int height = 2000
    );
};

#endif // MAP_RENDERER_HPP