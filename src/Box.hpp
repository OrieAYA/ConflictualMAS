#ifndef BOX_HPP
#define BOX_HPP

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <osmium/osm/location.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/osm/way.hpp>
#include <osmium/osm/box.hpp>
#include <osmium/handler.hpp>

// Structures de données
struct MyData {
    struct Point {
        double lat;
        double lon;
        osmium::object_id_type id = 0;
        std::vector<osmium::object_id_type> incident_ways;
        
        Point() = default;
        Point(double lat, double lon, osmium::object_id_type id = 0) 
            : lat(lat), lon(lon), id(id) {}
    };
    
    struct Way {
        osmium::object_id_type id;
        std::vector<Point> points;
        int weight = 0;
        float distance_meters = 0.0f;  // Distance totale du way en mètres
        
        Way() : id(0) {}
        Way(osmium::object_id_type id) : id(id) {}
    };

    std::unordered_map<osmium::object_id_type, Point> nodes;
    std::unordered_map<osmium::object_id_type, Way> ways;
    
    std::vector<Point> get_nodes_vector() const {
        std::vector<Point> result;
        result.reserve(nodes.size());
        for (const auto& [id, point] : nodes) {
            result.push_back(point);
        }
        return result;
    }
    
    std::vector<Way> get_ways_vector() const {
        std::vector<Way> result;
        result.reserve(ways.size());
        for (const auto& [id, way] : ways) {
            result.push_back(way);
        }
        return result;
    }
};

// Handler OSM
class MyHandler : public osmium::handler::Handler {
private:
    bool m_use_bbox_filter = false;
    std::unordered_map<osmium::object_id_type, osmium::Location> m_node_locations;
    
    // Fonction pour calculer la distance haversine
    double calculate_haversine_distance(double lat1, double lon1, double lat2, double lon2) const;

public:
    osmium::Box Map_bbox;
    MyData data_collector;

    MyHandler() {}
    
    void set_bounding_box(double min_lon, double min_lat, double max_lon, double max_lat) {
        Map_bbox = osmium::Box();
        Map_bbox.extend(osmium::Location(min_lon, min_lat));
        Map_bbox.extend(osmium::Location(max_lon, max_lat));
    }
    
    void enable_bounding_box_filter(bool enable = true) {
        m_use_bbox_filter = enable;
    }
    
    void node(const osmium::Node& node) {
        const osmium::Location node_loc = node.location();
        if (!node_loc.valid()) return;
        
        if (m_use_bbox_filter) {
            m_node_locations[node.id()] = node_loc;
            if (!Map_bbox.contains(node_loc)) {
                return;
            }
        }
        
        data_collector.nodes[node.id()] = MyData::Point(
            node_loc.lat(), 
            node_loc.lon(), 
            node.id()
        );
    }
    
    void way(const osmium::Way& way) {
        if (way.nodes().empty()) return;
        
        MyData::Way current_way(way.id());
        bool has_nodes_in_bbox = false;
        
        if (m_use_bbox_filter) {
            bool intersects_bbox = false;
            for (const auto& node_ref : way.nodes()) {
                auto it = m_node_locations.find(node_ref.ref());
                if (it != m_node_locations.end()) {
                    if (Map_bbox.contains(it->second)) {
                        intersects_bbox = true;
                        break;
                    }
                }
            }
            if (!intersects_bbox) return;
        }
        
        for (const auto& node_ref : way.nodes()) {
            auto it = data_collector.nodes.find(node_ref.ref());
            if (it != data_collector.nodes.end()) {
                current_way.points.push_back(it->second);
                has_nodes_in_bbox = true;
            }
        }
        
        if (has_nodes_in_bbox && current_way.points.size() > 1) {
            // Calculer la distance totale du way
            float total_distance = 0.0f;
            for (size_t i = 1; i < current_way.points.size(); ++i) {
                total_distance += static_cast<float>(calculate_haversine_distance(
                    current_way.points[i-1].lat, current_way.points[i-1].lon,
                    current_way.points[i].lat, current_way.points[i].lon
                ));
            }
            current_way.distance_meters = total_distance;
            
            data_collector.ways[way.id()] = current_way;
        }
    }
};

struct GeoBox {
    MyData data;
    osmium::Box bbox;
    std::string source_file;
    bool is_valid = false;
    
    GeoBox() = default;
    GeoBox(const MyData& d, const osmium::Box& b, const std::string& src) 
        : data(d), bbox(b), source_file(src), is_valid(true) {}
};

// Fonction indépendante pour créer une box depuis un fichier OSM
GeoBox create_geo_box(const std::string& osm_filename, 
                      double min_lon, double min_lat, 
                      double max_lon, double max_lat);

#endif // BOX_HPP