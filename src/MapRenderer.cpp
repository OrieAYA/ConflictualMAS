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

MapRenderer::MapRenderer() {
    // Initialize Mapnik if needed
}

void MapRenderer::render_shibuya_map(
    const MyData& data_collector,
    const osmium::Box& map_bbox,
    const std::string& output_filename,
    int width,
    int height
) {
    std::cout << "Rendering Shibuya map..." << std::endl;

    mapnik::Map m(width, height);
    m.set_background(mapnik::color("white"));

    // Create memory datasource for nodes
    mapnik::parameters node_params;
    node_params["type"] = "memory";
    auto node_ds = std::make_shared<mapnik::memory_datasource>(node_params);
    
    for (const auto& [node_id, point_data] : data_collector.nodes) {
        mapnik::context_ptr ctx = std::make_shared<mapnik::context_type>();
        auto feature = std::make_shared<mapnik::feature_impl>(ctx, node_id);
        
        // Create point geometry
        mapnik::geometry::point<double> point_geom(point_data.lon, point_data.lat);
        feature->set_geometry(std::move(point_geom));
        
        node_ds->push(feature);
    }
    std::cout << "Added " << data_collector.nodes.size() << " nodes to datasource." << std::endl;

    // Create memory datasource for ways
    mapnik::parameters way_params;
    way_params["type"] = "memory";
    auto way_ds = std::make_shared<mapnik::memory_datasource>(way_params);
    
    for (const auto& [way_id, way_data] : data_collector.ways) {
        if (way_data.points.size() < 2) continue;
        
        mapnik::context_ptr ctx = std::make_shared<mapnik::context_type>();
        auto feature = std::make_shared<mapnik::feature_impl>(ctx, way_id);
        
        // Create line string geometry
        mapnik::geometry::line_string<double> line_geom;
        for (const auto& point : way_data.points) {
            line_geom.emplace_back(point.lon, point.lat);
        }
        
        feature->set_geometry(std::move(line_geom));
        way_ds->push(feature);
    }
    std::cout << "Added " << data_collector.ways.size() << " ways to datasource." << std::endl;

    // Define styles using modern symbolizer properties
    mapnik::feature_type_style node_style;
    mapnik::rule point_rule;
    mapnik::markers_symbolizer point_sym;
    
    // Set marker properties using put() method
    mapnik::put(point_sym, mapnik::keys::fill, mapnik::color("red"));
    mapnik::put(point_sym, mapnik::keys::width, mapnik::value_double(8.0));
    mapnik::put(point_sym, mapnik::keys::height, mapnik::value_double(8.0));
    
    point_rule.append(std::move(point_sym));
    node_style.add_rule(std::move(point_rule));

    mapnik::feature_type_style way_style;
    mapnik::rule line_rule;
    mapnik::line_symbolizer line_sym;
    
    // Set line properties using put() method
    mapnik::put(line_sym, mapnik::keys::stroke, mapnik::color("blue"));
    mapnik::put(line_sym, mapnik::keys::stroke_width, mapnik::value_double(2.0));
    
    line_rule.append(std::move(line_sym));
    way_style.add_rule(std::move(line_rule));

    m.insert_style("nodes_style", std::move(node_style));
    m.insert_style("ways_style", std::move(way_style));

    // Create and add layers
    mapnik::layer way_layer("ways");
    way_layer.set_datasource(way_ds);
    way_layer.add_style("ways_style");
    m.add_layer(way_layer);

    mapnik::layer node_layer("nodes");
    node_layer.set_datasource(node_ds);
    node_layer.add_style("nodes_style");
    m.add_layer(node_layer);

    // Set map extent
    m.zoom_to_box(mapnik::box2d<double>(
        map_bbox.bottom_left().lon(),
        map_bbox.bottom_left().lat(),
        map_bbox.top_right().lon(),
        map_bbox.top_right().lat()
    ));

    // Render the map
    try {
        mapnik::image_rgba8 im(m.width(), m.height());
        mapnik::agg_renderer<mapnik::image_rgba8> rend(m, im);
        rend.apply();

        mapnik::save_to_file(im, output_filename);
        std::cout << "Map rendered to " << output_filename << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Rendering error: " << e.what() << std::endl;
        throw;
    }
}