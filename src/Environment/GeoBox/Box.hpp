#ifndef BOX_HPP
#define BOX_HPP

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <map>
#include <osmium/osm/location.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/osm/way.hpp>
#include <osmium/osm/box.hpp>
#include <osmium/handler.hpp>

// Fonctions de validation
bool is_valid_way_type(const osmium::Way& way);

// Configuration pour l'API Flickr
struct FlickrConfig {
    std::string api_key;
    std::string search_word;
    std::string bbox;
    std::string min_date;
    std::string max_date;
    double poi_assignment_radius = 100.0;
    
    FlickrConfig() = default;
    FlickrConfig(const std::string& key, const std::string& word, const std::string& bounding_box)
        : api_key(key), search_word(word), bbox(bounding_box) {}
};

// Structure pour un Point d'Intérêt Flickr
struct FlickrPOI {
    std::string id;
    double latitude;
    double longitude;
    std::string title;
    std::string description;
    std::vector<std::string> tags;
    
    FlickrPOI() = default;
};

// Groupe d'objectifs
struct ObjectiveGroup {
    int id;
    std::string name;
    std::string description;
    int point_count = 0;
    std::vector<osmium::object_id_type> node_ids;
    
    ObjectiveGroup() = default;
    ObjectiveGroup(int id, const std::string& name, const std::string& desc)
        : id(id), name(name), description(desc) {}
};

// Client API Flickr
class FlickrAPIClient {
private:
    FlickrConfig config;
    
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response);
    std::string build_flickr_url(const std::string& method, const std::map<std::string, std::string>& params) const;
    std::string make_http_request(const std::string& url) const;
    
public:
    FlickrAPIClient(const FlickrConfig& cfg) : config(cfg) {}
    
    std::vector<FlickrPOI> fetch_pois() const;
    void save_pois_to_file(const std::vector<FlickrPOI>& pois, const std::string& filename) const;
    std::vector<FlickrPOI> load_pois_from_file(const std::string& filename) const;
};

// Fonction utilitaire pour calculer la distance haversine
double calculate_haversine_distance(double lat1, double lon1, double lat2, double lon2);

// Structure de données principale
struct MyData {
    struct Point {
        double lat;
        double lon;
        osmium::object_id_type id = 0;
        std::vector<osmium::object_id_type> incident_ways;
        
        std::unordered_set<int> groupes;  // Remplace int groupe
        std::string objective_id = "";
        
        Point() = default;
        Point(double lat, double lon, osmium::object_id_type id = 0) 
            : lat(lat), lon(lon), id(id) {}
            
        // Méthodes utilitaires pour compatibilité
        int get_primary_group() const {
            return groupes.empty() ? 0 : *groupes.begin();
        }
        
        void set_group(int group) {
            groupes.clear();
            if (group != 0) groupes.insert(group);
        }
        
        void add_group(int group) {
            if (group != 0) groupes.insert(group);
        }
        
        void remove_group(int group) {
            groupes.erase(group);
        }
        
        bool has_group(int group) const {
            return groupes.count(group) > 0;
        }
    };
    
    struct Way {
        osmium::object_id_type id;
        osmium::object_id_type node1_id;
        osmium::object_id_type node2_id;
        std::vector<Point> points;
        float distance_meters = 0.0f;
        
        std::unordered_set<int> groupes;  // Remplace int groupe
        
        Way() : id(0), node1_id(0), node2_id(0) {}
        Way(osmium::object_id_type id) : id(id), node1_id(0), node2_id(0) {}
        Way(osmium::object_id_type id, osmium::object_id_type n1, osmium::object_id_type n2) 
            : id(id), node1_id(n1), node2_id(n2) {}
            
        // Méthodes utilitaires pour compatibilité
        int get_primary_group() const {
            return groupes.empty() ? 0 : *groupes.begin();
        }
        
        void set_group(int group) {
            groupes.clear();
            if (group != 0) groupes.insert(group);
        }
        
        void add_group(int group) {
            if (group != 0) groupes.insert(group);
        }
        
        void remove_group(int group) {
            groupes.erase(group);
        }
        
        bool has_group(int group) const {
            return groupes.count(group) > 0;
        }
    };

    std::unordered_map<osmium::object_id_type, Point> nodes;
    std::unordered_map<osmium::object_id_type, Way> ways;
    std::unordered_map<int, ObjectiveGroup> objective_groups;
    
    // Méthode pour afficher les groupes d'objectifs
    void print_objective_groups() const {
        std::cout << "\n=== Groupes d'objectifs ===" << std::endl;
        for (const auto& [id, group] : objective_groups) {
            std::cout << "Groupe " << id << ": " << group.name 
                      << " (" << group.point_count << " points)" << std::endl;
            std::cout << "  Description: " << group.description << std::endl;
        }
    }
};

// Handler OSM optimisé
class MyHandler : public osmium::handler::Handler {
private:
    bool m_use_bbox_filter = false;
    std::unordered_map<osmium::object_id_type, osmium::Location> m_node_locations;
    
    double calculate_haversine_distance(double lat1, double lon1, double lat2, double lon2) const;
    
    static osmium::object_id_type generate_new_way_id(osmium::object_id_type base_id, size_t segment_index) {
        return base_id * 1000000 + segment_index;
    }
    
    // Helper pour connecter un way à ses nodes
    void connect_way_to_nodes(osmium::object_id_type way_id, 
                             osmium::object_id_type node1_id, 
                             osmium::object_id_type node2_id);

public:
    osmium::Box Map_bbox;
    MyData data_collector;

    MyHandler() {}
    
    void set_bounding_box(double min_lon, double min_lat, double max_lon, double max_lat);
    void enable_bounding_box_filter(bool enable = true) { m_use_bbox_filter = enable; }
    
    void node(const osmium::Node& node);
    void way(const osmium::Way& way);
};

// Structure GeoBox
struct GeoBox {
    MyData data;
    osmium::Box bbox;
    std::string source_file;
    bool is_valid = false;
    
    GeoBox() = default;
    GeoBox(const MyData& d, const osmium::Box& b, const std::string& src) 
        : data(d), bbox(b), source_file(src), is_valid(true) {}
};

// Fonctions principales
GeoBox create_geo_box(const std::string& osm_filename, 
                      double min_lon, double min_lat, 
                      double max_lon, double max_lat);

GeoBox apply_objectives(GeoBox geo_box, const FlickrConfig& flickr_config, 
                       const std::string& cache_filename, bool use_cache = true, int group_id = 1);

// Fonctions utilitaires pour les composantes
void bfs_explore(const MyData& data, osmium::object_id_type start,
                std::unordered_set<osmium::object_id_type>& visited,
                std::vector<osmium::object_id_type>& component);

std::vector<std::vector<osmium::object_id_type>> find_components_simple(const MyData& data);

osmium::object_id_type get_max_way_id(const MyData& data);

std::tuple<osmium::object_id_type, osmium::object_id_type, double> 
find_closest_nodes(const MyData& data, 
                  const std::vector<osmium::object_id_type>& comp1,
                  const std::vector<osmium::object_id_type>& comp2);

void create_connecting_way(MyData& data, osmium::object_id_type way_id, 
                          osmium::object_id_type node1_id, osmium::object_id_type node2_id, 
                          double distance);

GeoBox connect_isolated_components(GeoBox geo_box);

#endif // BOX_HPP