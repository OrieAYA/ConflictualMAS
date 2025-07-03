#ifndef MAP_RENDERER_HPP
#define MAP_RENDERER_HPP

// Assurez-vous que MyData est défini dans Box.hpp
#include "Box.hpp"
// Nécessaire pour osmium::Box si vous l'utilisez dans la signature de la fonction membre
#include <osmium/osm/box.hpp>
// Nécessaire pour std::string dans la signature
#include <string>

class MapRenderer {
public:
    MapRenderer();

    // Déclaration de la fonction membre render_shibuya_map
    // J'ai inclus les valeurs par défaut ici, car c'est la déclaration.
    void render_shibuya_map(
        const MyData& data_collector,
        const osmium::Box& map_bbox,
        const std::string& output_filename = "shibuya_map.png",
        int width = 2000,
        int height = 2000
    );
};

#endif // MAP_RENDERER_HPP