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

// Fonction indépendante : Rendu depuis MyData - CORRIGÉE
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
        for (int group : point_data.groupes) {
            node_group_counts[group]++;
        }
        if (point_data.groupes.empty()) {
            node_group_counts[0]++;
        }
    }
    
    for (const auto& [way_id, way_data] : data.ways) {
        for (int group : way_data.groupes) {
            way_group_counts[group]++;
        }
        if (way_data.groupes.empty()) {
            way_group_counts[0]++;
        }
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
            
            // Logique simplifiée pour déterminer la couleur des POI
            if (point_data.groupes.empty()) {
                // Groupe 0 uniquement - noir normal
                feature->put("groupe", 0);
            } else if (point_data.groupes.size() == 1) {
                // Un seul groupe non-zéro - couleur du groupe
                feature->put("groupe", *point_data.groupes.begin());
            } else if (point_data.groupes.size() == 2) {
                // 2 groupes (dont 0) - couleur du groupe non-zéro
                int non_zero_group = 0;
                for (int group : point_data.groupes) {
                    if (group != 0) {
                        non_zero_group = group;
                        break;
                    }
                }
                feature->put("groupe", non_zero_group);
            } else {
                // 3 groupes ou plus - vert foncé (code spécial 99)
                feature->put("groupe", 99);
            }
            
            node_ds->push(feature);
        }
        std::cout << "Added " << data.nodes.size() << " nodes to datasource." << std::endl;

        // === CRÉER DATASOURCE POUR LES WAYS ===
        mapnik::parameters way_params;
        way_params["type"] = "memory";
        auto way_ds = std::make_shared<mapnik::memory_datasource>(way_params);
        
        int ways_added = 0;
        int ways_skipped = 0;
        
        for (const auto& [way_id, way_data] : data.ways) {
            auto node1_it = data.nodes.find(way_data.node1_id);
            auto node2_it = data.nodes.find(way_data.node2_id);
            
            if (node1_it == data.nodes.end()) {
                ways_skipped++;
                std::cerr << "ERROR: Way " << way_id << " references missing node1: " 
                          << way_data.node1_id << std::endl;
                continue;
            }
            
            if (node2_it == data.nodes.end()) {
                ways_skipped++;
                std::cerr << "ERROR: Way " << way_id << " references missing node2: " 
                          << way_data.node2_id << std::endl;
                continue;
            }
            
            if (way_data.node1_id == way_data.node2_id) {
                ways_skipped++;
                std::cerr << "ERROR: Way " << way_id << " has identical nodes: " 
                          << way_data.node1_id << std::endl;
                continue;
            }
            
            mapnik::context_ptr ctx = std::make_shared<mapnik::context_type>();
            ctx->push("groupe");
            auto feature = std::make_shared<mapnik::feature_impl>(ctx, way_id);
            
            mapnik::geometry::line_string<double> line_geom;
            line_geom.emplace_back(node1_it->second.lon, node1_it->second.lat);
            line_geom.emplace_back(node2_it->second.lon, node2_it->second.lat);
            feature->set_geometry(std::move(line_geom));
            
            // Logique simplifiée pour déterminer la couleur des ways
            if (way_data.groupes.empty()) {
                // Groupe 0 uniquement - gris normal
                feature->put("groupe", 0);
            } else if (way_data.groupes.size() == 1) {
                // Un seul groupe non-zéro - couleur du groupe
                feature->put("groupe", *way_data.groupes.begin());
            } else if (way_data.groupes.size() == 2) {
                // 2 groupes (dont 0) - couleur du groupe non-zéro
                int non_zero_group = 0;
                for (int group : way_data.groupes) {
                    if (group != 0) {
                        non_zero_group = group;
                        break;
                    }
                }
                feature->put("groupe", non_zero_group);
            } else {
                // 3 groupes ou plus - vert foncé (code spécial 99)
                feature->put("groupe", 99);
            }
            
            way_ds->push(feature);
            ways_added++;
        }
        
        std::cout << "Ways processing results:" << std::endl;
        std::cout << "  Added: " << ways_added << " ways" << std::endl;
        std::cout << "  Skipped: " << ways_skipped << " ways" << std::endl;

        // === STYLES POUR LES NODES ===
        mapnik::feature_type_style node_style;

        // Style pour les nodes normaux (groupe 0)
        mapnik::rule normal_point_rule;
        normal_point_rule.set_filter(mapnik::parse_expression("[groupe] = 0"));
        mapnik::markers_symbolizer normal_point_sym;
        
        mapnik::put(normal_point_sym, mapnik::keys::fill, mapnik::color("black"));
        mapnik::put(normal_point_sym, mapnik::keys::width, mapnik::value_double(3.0));
        mapnik::put(normal_point_sym, mapnik::keys::height, mapnik::value_double(3.0));
        
        normal_point_rule.append(std::move(normal_point_sym));
        node_style.add_rule(std::move(normal_point_rule));

        // Style pour les POI avec 3+ groupes (vert foncé)
        mapnik::rule multi_poi_rule;
        multi_poi_rule.set_filter(mapnik::parse_expression("[groupe] = 99"));
        mapnik::markers_symbolizer multi_poi_sym;
        
        mapnik::put(multi_poi_sym, mapnik::keys::fill, mapnik::color("#006600")); // Vert foncé
        mapnik::put(multi_poi_sym, mapnik::keys::stroke, mapnik::color("white"));
        mapnik::put(multi_poi_sym, mapnik::keys::stroke_width, mapnik::value_double(1.0));
        mapnik::put(multi_poi_sym, mapnik::keys::width, mapnik::value_double(8.0));
        mapnik::put(multi_poi_sym, mapnik::keys::height, mapnik::value_double(8.0));
        
        multi_poi_rule.append(std::move(multi_poi_sym));
        node_style.add_rule(std::move(multi_poi_rule));

        // Palette de couleurs
        std::vector<std::string> poi_colors = {
            "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",  // Rouge, Bleu, Vert, Orange, Violet
            "#1ABC9C", "#34495E", "#E67E22", "#8E44AD", "#C0392B"   // Turquoise, Gris foncé, Orange foncé, Violet foncé, Rouge foncé
        };

        // Styles pour les POI par groupe
        for (int group_id = 1; group_id <= 10; ++group_id) {
            mapnik::rule poi_rule;
            poi_rule.set_filter(mapnik::parse_expression("[groupe] = " + std::to_string(group_id)));
            
            mapnik::markers_symbolizer poi_sym;
            std::string color = poi_colors[(group_id - 1) % poi_colors.size()];
            mapnik::put(poi_sym, mapnik::keys::fill, mapnik::color(color));
            mapnik::put(poi_sym, mapnik::keys::stroke, mapnik::color("white"));
            mapnik::put(poi_sym, mapnik::keys::stroke_width, mapnik::value_double(1.0));
            mapnik::put(poi_sym, mapnik::keys::width, mapnik::value_double(8.0));
            mapnik::put(poi_sym, mapnik::keys::height, mapnik::value_double(8.0));
            
            poi_rule.append(std::move(poi_sym));
            node_style.add_rule(std::move(poi_rule));
        }

        // === STYLES POUR LES WAYS ===
        mapnik::feature_type_style way_style;
        
        // Style pour les ways normaux (groupe 0)
        mapnik::rule normal_way_rule;
        normal_way_rule.set_filter(mapnik::parse_expression("[groupe] = 0"));
        mapnik::line_symbolizer normal_way_sym;
        
        mapnik::put(normal_way_sym, mapnik::keys::stroke, mapnik::color("grey"));
        mapnik::put(normal_way_sym, mapnik::keys::stroke_width, mapnik::value_double(1.0));
        mapnik::put(normal_way_sym, mapnik::keys::stroke_opacity, mapnik::value_double(0.8));
        
        normal_way_rule.append(std::move(normal_way_sym));
        way_style.add_rule(std::move(normal_way_rule));

        // Style pour les ways avec 3+ groupes (vert foncé)
        mapnik::rule multi_way_rule;
        multi_way_rule.set_filter(mapnik::parse_expression("[groupe] = 99"));
        mapnik::line_symbolizer multi_way_sym;
        
        mapnik::put(multi_way_sym, mapnik::keys::stroke, mapnik::color("#006600")); // Vert foncé
        mapnik::put(multi_way_sym, mapnik::keys::stroke_width, mapnik::value_double(4.0));
        mapnik::put(multi_way_sym, mapnik::keys::stroke_opacity, mapnik::value_double(0.9));
        mapnik::put(multi_way_sym, mapnik::keys::stroke_linecap, mapnik::line_cap_enum::ROUND_CAP);
        
        multi_way_rule.append(std::move(multi_way_sym));
        way_style.add_rule(std::move(multi_way_rule));
        
        // Styles pour les ways de pathfinding (groupes 1-10)
        std::vector<std::string> path_colors = {
            "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",  // Rouge, Bleu, Vert, Orange, Violet
            "#1ABC9C", "#34495E", "#E67E22", "#8E44AD", "#C0392B"   // Turquoise, Gris foncé, Orange foncé, Violet foncé, Rouge foncé
        };
        
        for (int group_id = 1; group_id <= 10; ++group_id) {
            mapnik::rule path_rule;
            path_rule.set_filter(mapnik::parse_expression("[groupe] = " + std::to_string(group_id)));
            
            mapnik::line_symbolizer path_sym;
            std::string color = path_colors[(group_id - 1) % path_colors.size()];
            mapnik::put(path_sym, mapnik::keys::stroke, mapnik::color(color));
            mapnik::put(path_sym, mapnik::keys::stroke_width, mapnik::value_double(4.0));
            mapnik::put(path_sym, mapnik::keys::stroke_opacity, mapnik::value_double(0.9));
            mapnik::put(path_sym, mapnik::keys::stroke_linecap, mapnik::line_cap_enum::ROUND_CAP);
            
            path_rule.append(std::move(path_sym));
            way_style.add_rule(std::move(path_rule));
        }

        // === AJOUTER LES STYLES ===
        m.insert_style("nodes_style", std::move(node_style));
        m.insert_style("ways_style", std::move(way_style));

        // === AJOUTER LES LAYERS ===
        mapnik::layer way_layer("ways");
        way_layer.set_datasource(way_ds);
        way_layer.add_style("ways_style");
        m.add_layer(way_layer);

        mapnik::layer node_layer("nodes");
        node_layer.set_datasource(node_ds);
        node_layer.add_style("nodes_style");
        m.add_layer(node_layer);

        // === DÉFINIR L'ÉTENDUE ===
        m.zoom_to_box(mapnik::box2d<double>(
            bbox.bottom_left().lon(),
            bbox.bottom_left().lat(),
            bbox.top_right().lon(),
            bbox.top_right().lat()
        ));

        // === RENDU ===
        mapnik::image_rgba8 im(m.width(), m.height());
        mapnik::agg_renderer<mapnik::image_rgba8> rend(m, im);
        rend.apply();

        // Sauvegarde avec gestion des doublons
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