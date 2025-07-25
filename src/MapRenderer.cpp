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
#include <map>
#include <numeric>

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

// Fonction indépendante : Rendu depuis MyData avec RÉCUPÉRATION CORRECTE DES COORDONNÉES
bool render_map_from_data(const MyData& data,
                         const osmium::Box& bbox,
                         const std::string& output_filename,
                         int width,
                         int height) {
    
    std::cout << "Rendering map from data..." << std::endl;

    // Debug: Compter les points et ways par groupe
    std::map<int, int> node_group_counts;
    std::map<int, int> way_group_counts;
    
    for (const auto& [node_id, point_data] : data.nodes) {
        node_group_counts[point_data.groupe]++;
    }
    
    for (const auto& [way_id, way_data] : data.ways) {
        way_group_counts[way_data.groupe]++;
    }

    std::cout << "Points par groupe:" << std::endl;
    for (const auto& [group_id, count] : node_group_counts) {
        std::cout << "  Groupe " << group_id << ": " << count << " points" << std::endl;
    }
    
    std::cout << "Ways par groupe:" << std::endl;
    for (const auto& [group_id, count] : way_group_counts) {
        std::cout << "  Groupe " << group_id << ": " << count << " ways" << std::endl;
    }
    
    try {
        mapnik::Map m(width, height);
        m.set_background(mapnik::color("white"));

        // === CRÉER DATASOURCE POUR LES NODES ===
        mapnik::parameters node_params;
        node_params["type"] = "memory";
        auto node_ds = std::make_shared<mapnik::memory_datasource>(node_params);

        for (const auto& [node_id, point_data] : data.nodes) {
            mapnik::context_ptr ctx = std::make_shared<mapnik::context_type>();
            ctx->push("groupe");
            auto feature = std::make_shared<mapnik::feature_impl>(ctx, node_id);
            
            mapnik::geometry::point<double> point_geom(point_data.lon, point_data.lat);
            feature->set_geometry(std::move(point_geom));
            
            feature->put("groupe", point_data.groupe);
            
            node_ds->push(feature);
        }
        std::cout << "Added " << data.nodes.size() << " nodes to datasource." << std::endl;

        // === CRÉER DATASOURCE POUR LES WAYS - CORRECTION MAJEURE ===
        mapnik::parameters way_params;
        way_params["type"] = "memory";
        auto way_ds = std::make_shared<mapnik::memory_datasource>(way_params);
        
        int ways_added = 0;
        int ways_skipped = 0;
        
        for (const auto& [way_id, way_data] : data.ways) {
            // CORRECTIF: Récupérer les coordonnées depuis data.nodes
            auto node1_it = data.nodes.find(way_data.node1_id);
            auto node2_it = data.nodes.find(way_data.node2_id);
            
            if (node1_it == data.nodes.end() || node2_it == data.nodes.end()) {
                ways_skipped++;
                std::cerr << "Way " << way_id << ": nodes introuvables (" 
                          << way_data.node1_id << ", " << way_data.node2_id << ")" << std::endl;
                continue;
            }
            
            mapnik::context_ptr ctx = std::make_shared<mapnik::context_type>();
            ctx->push("groupe");
            ctx->push("distance");
            auto feature = std::make_shared<mapnik::feature_impl>(ctx, way_id);
            
            // CORRECTIF: Construire la géométrie avec les vraies coordonnées
            mapnik::geometry::line_string<double> line_geom;
            line_geom.emplace_back(node1_it->second.lon, node1_it->second.lat);
            line_geom.emplace_back(node2_it->second.lon, node2_it->second.lat);
            
            feature->set_geometry(std::move(line_geom));
            feature->put("groupe", way_data.groupe);
            feature->put("distance", way_data.distance_meters);
            
            way_ds->push(feature);
            ways_added++;
        }
        
        std::cout << "Ways processing results:" << std::endl;
        std::cout << "  Added: " << ways_added << " ways" << std::endl;
        std::cout << "  Skipped: " << ways_skipped << " ways (missing nodes)" << std::endl;

        // === DÉFINIR LES STYLES POUR LES NODES ===
        mapnik::feature_type_style node_style;

        // Style pour les nodes non-objectifs (groupe 0)
        mapnik::rule normal_point_rule;
        normal_point_rule.set_filter(mapnik::parse_expression("[groupe] = 0"));
        mapnik::markers_symbolizer normal_point_sym;
        
        mapnik::put(normal_point_sym, mapnik::keys::fill, mapnik::color("black"));
        mapnik::put(normal_point_sym, mapnik::keys::width, mapnik::value_double(4.0));
        mapnik::put(normal_point_sym, mapnik::keys::height, mapnik::value_double(4.0));
        
        normal_point_rule.append(std::move(normal_point_sym));
        node_style.add_rule(std::move(normal_point_rule));

        // Styles pour les nodes objectifs (groupes 1-5) avec couleurs distinctes
        std::vector<std::string> objective_colors = {"red", "blue", "green", "orange", "purple"};
        
        for (int group_id = 1; group_id <= 5; ++group_id) {
            mapnik::rule objective_rule;
            objective_rule.set_filter(mapnik::parse_expression("[groupe] = " + std::to_string(group_id)));
            
            mapnik::markers_symbolizer objective_sym;
            std::string color = objective_colors[(group_id - 1) % objective_colors.size()];
            mapnik::put(objective_sym, mapnik::keys::fill, mapnik::color(color));
            mapnik::put(objective_sym, mapnik::keys::width, mapnik::value_double(6.0));
            mapnik::put(objective_sym, mapnik::keys::height, mapnik::value_double(6.0));
            
            objective_rule.append(std::move(objective_sym));
            node_style.add_rule(std::move(objective_rule));
        }

        // === DÉFINIR LES STYLES POUR LES WAYS ===
        mapnik::feature_type_style way_style;
        
        // Style pour les ways normaux (groupe 0)
        mapnik::rule normal_way_rule;
        normal_way_rule.set_filter(mapnik::parse_expression("[groupe] = 0"));
        mapnik::line_symbolizer normal_way_sym;
        
        mapnik::put(normal_way_sym, mapnik::keys::stroke, mapnik::color("grey"));
        mapnik::put(normal_way_sym, mapnik::keys::stroke_width, mapnik::value_double(2.0));
        
        normal_way_rule.append(std::move(normal_way_sym));
        way_style.add_rule(std::move(normal_way_rule));
        
        // Styles pour les ways de chemins (groupes 1-5) - Plus épais et colorés
        std::vector<std::string> path_colors = {"#FF0066", "#0066FF", "#00CC66", "#FF9900", "#9900CC"};
        
        for (int group_id = 1; group_id <= 5; ++group_id) {
            mapnik::rule path_rule;
            path_rule.set_filter(mapnik::parse_expression("[groupe] = " + std::to_string(group_id)));
            
            mapnik::line_symbolizer path_sym;
            std::string color = path_colors[(group_id - 1) % path_colors.size()];
            mapnik::put(path_sym, mapnik::keys::stroke, mapnik::color(color));
            mapnik::put(path_sym, mapnik::keys::stroke_width, mapnik::value_double(2.0));
            mapnik::put(path_sym, mapnik::keys::stroke_opacity, mapnik::value_double(1.0));
            
            path_rule.append(std::move(path_sym));
            way_style.add_rule(std::move(path_rule));
        }

        // === AJOUTER LES STYLES À LA CARTE ===
        m.insert_style("nodes_style", std::move(node_style));
        m.insert_style("ways_style", std::move(way_style));

        // === CRÉER ET AJOUTER LES LAYERS ===
        // Layer ways (en arrière-plan)
        mapnik::layer way_layer("ways");
        way_layer.set_datasource(way_ds);
        way_layer.add_style("ways_style");
        m.add_layer(way_layer);

        // Layer nodes (au premier plan)
        mapnik::layer node_layer("nodes");
        node_layer.set_datasource(node_ds);
        node_layer.add_style("nodes_style");
        m.add_layer(node_layer);

        // === DÉFINIR L'ÉTENDUE DE LA CARTE ===
        m.zoom_to_box(mapnik::box2d<double>(
            bbox.bottom_left().lon(),
            bbox.bottom_left().lat(),
            bbox.top_right().lon(),
            bbox.top_right().lat()
        ));

        // === RENDU ET SAUVEGARDE ===
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
        
        std::cout << "Rendu terminé:" << std::endl;
        std::cout << "  - Chemins tracés (groupes > 0): ";
        
        int traced_ways = 0;
        int objective_points = 0;
        
        for (const auto& [group_id, count] : way_group_counts) {
            if (group_id > 0) traced_ways += count;
        }
        
        for (const auto& [group_id, count] : node_group_counts) {
            if (group_id > 0) objective_points += count;
        }
        
        std::cout << traced_ways << std::endl;
        std::cout << "  - Points objectifs (groupes > 0): " << objective_points << std::endl;
        
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Rendering error: " << e.what() << std::endl;
        return false;
    }
}