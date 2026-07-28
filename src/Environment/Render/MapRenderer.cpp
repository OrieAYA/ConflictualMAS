#include "MapRenderer.hpp"
#include <iostream>

// Rendu PNG desactive : dependance Mapnik retiree du build. L'interface est
// conservee pour que GeoBoxManager::render_geobox et le menu Legacy compilent
// et lient sans modification. Renderer fonctionnel sur la branche git `mapnik`.

namespace {
bool render_disabled(const std::string& output_filename) {
    std::cerr << "[render] desactive (build sans Mapnik) - '"
              << output_filename << "' non genere.\n";
    return false;
}
}

bool render_map(const GeoBox& geo_box,
                const std::string& output_filename,
                int width,
                int height) {
    (void)geo_box; (void)width; (void)height;
    return render_disabled(output_filename);
}

bool render_map_from_data(const MyData& data,
                          const osmium::Box& bbox,
                          const std::string& output_filename,
                          int width,
                          int height) {
    (void)data; (void)bbox; (void)width; (void)height;
    return render_disabled(output_filename);
}
