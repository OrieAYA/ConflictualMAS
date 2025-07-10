#include "MapRenderer.hpp"
#include <mapnik/map.hpp>
#include <mapnik/layer.hpp>
#include <mapnik/rule.hpp>
#include <mapnik/feature_type_style.hpp>
#include <mapnik/agg_renderer.hpp>
#include <mapnik/image_util.hpp>
#include <mapnik/datasource_cache.hpp>
#include <mapnik/memory_datasource.hpp>
#include <mapnik/geometry.hpp>
#include <mapnik/symbolizer.hpp>
#include <mapnik/color.hpp>
#include <mapnik/feature.hpp>
#include <mapnik/value.hpp>
#include <mapnik/params.hpp>
#include <iostream>
#include <filesystem>

// Fonction indépendante : Rendu depuis GeoBox
bool render_map(const GeoBox& geo_box,
                const std::string& output_filename,
                int width,
                int height) {
    
    if (!geo_box.is_valid) {
        std::cerr << "Cannot render: GeoBox is invalid" << std::endl;
        return false;
    }
    
    std::cout << "Rendering map from GeoBox..." << std::endl;
    std::cout << "Source: " << geo_box.source_file << std::endl;
    std::cout << "Nodes: " << geo_box.data.nodes.size() << std::endl;
    std::cout << "Ways: " << geo_box.data.ways.size() << std::endl;
    
    return render_map_from_data(geo_box.data, geo_box.bbox, output_filename, width, height);
}

// Fonction indépendante : Rendu depuis MyData
bool render_map_from_data(const MyData& data,
                         const osmium::Box& bbox,
                         const std::string& output_filename,
                         int width,
                         int height) {
    
    std::cout << "Rendering map from data..." << std::endl;
    
    try {
        mapnik::Map m(width, height);
        m.set_background(mapnik::color("white"));

        // Créer datasource pour les nodes
        mapnik::parameters node_params;
        node_params["type"] = "memory";
        auto node_ds = std::make_shared<mapnik::memory_datasource>(node_params);
        
        for (const auto& [node_id, point_data] : data.nodes) {
            mapnik::context_ptr ctx = std::make_shared<mapnik::context_type>();
            auto feature = std::make_shared<mapnik::feature_impl>(ctx, node_id);
            
            mapnik::geometry::point<double> point_geom(point_data.lon, point_data.lat);
            feature->set_geometry(std::move(point_geom));
            
            node_ds->push(feature);
        }
        std::cout << "Added " << data.nodes.size() << " nodes to datasource." << std::endl;

        // Créer datasource pour les ways
        mapnik::parameters way_params;
        way_params["type"] = "memory";
        auto way_ds = std::make_shared<mapnik::memory_datasource>(way_params);
        
        for (const auto& [way_id, way_data] : data.ways) {
            if (way_data.points.size() < 2) continue;
            
            mapnik::context_ptr ctx = std::make_shared<mapnik::context_type>();
            auto feature = std::make_shared<mapnik::feature_impl>(ctx, way_id);
            
            mapnik::geometry::line_string<double> line_geom;
            for (const auto& point : way_data.points) {
                line_geom.emplace_back(point.lon, point.lat);
            }
            
            feature->set_geometry(std::move(line_geom));
            way_ds->push(feature);
        }
        std::cout << "Added " << data.ways.size() << " ways to datasource." << std::endl;

        // Définir les styles
        mapnik::feature_type_style node_style;
        mapnik::rule point_rule;
        mapnik::markers_symbolizer point_sym;
        
        mapnik::put(point_sym, mapnik::keys::fill, mapnik::color("black"));
        mapnik::put(point_sym, mapnik::keys::width, mapnik::value_double(6.0));
        mapnik::put(point_sym, mapnik::keys::height, mapnik::value_double(6.0));
        
        point_rule.append(std::move(point_sym));
        node_style.add_rule(std::move(point_rule));

        mapnik::feature_type_style way_style;
        mapnik::rule line_rule;
        mapnik::line_symbolizer line_sym;
        
        mapnik::put(line_sym, mapnik::keys::stroke, mapnik::color("grey"));
        mapnik::put(line_sym, mapnik::keys::stroke_width, mapnik::value_double(2.0));
        
        line_rule.append(std::move(line_sym));
        way_style.add_rule(std::move(line_rule));

        m.insert_style("nodes_style", std::move(node_style));
        m.insert_style("ways_style", std::move(way_style));

        // Créer et ajouter les layers
        mapnik::layer way_layer("ways");
        way_layer.set_datasource(way_ds);
        way_layer.add_style("ways_style");
        m.add_layer(way_layer);

        mapnik::layer node_layer("nodes");
        node_layer.set_datasource(node_ds);
        node_layer.add_style("nodes_style");
        m.add_layer(node_layer);

        // Définir l'étendue de la carte
        m.zoom_to_box(mapnik::box2d<double>(
            bbox.bottom_left().lon(),
            bbox.bottom_left().lat(),
            bbox.top_right().lon(),
            bbox.top_right().lat()
        ));

        // Rendu et sauvegarde
        mapnik::image_rgba8 im(m.width(), m.height());
        mapnik::agg_renderer<mapnik::image_rgba8> rend(m, im);
        rend.apply();

        // Gestion des noms de fichiers avec numérotation
        int counter = 0;
        namespace fs = std::filesystem;
        fs::path dir_path = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\maps\\";
        fs::path base_path = output_filename;
        std::string ext = ".png";

        if (base_path.has_extension()) {
            base_path = base_path.stem();
        }

        std::string final_output_filename = (dir_path / (base_path.string() + ext)).string();

        while (fs::exists(final_output_filename)) {
            counter++;
            final_output_filename = (dir_path / (base_path.string() + "(" + std::to_string(counter) + ")" + ext)).string();
        }

        mapnik::save_to_file(im, final_output_filename);
        std::cout << "Map successfully rendered to: " << final_output_filename << std::endl;
        
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Rendering error: " << e.what() << std::endl;
        return false;
    }
}