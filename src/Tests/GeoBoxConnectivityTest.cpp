#include "Tests/GeoBoxConnectivityTest.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include <iostream>
#include <queue>
#include <unordered_set>
#include <algorithm>

// BFS non-dirigé : même logique que keep_largest_component dans Box.cpp.
// Retourne le nombre de nœuds atteints depuis `start`.
static size_t bfs_count(const MyData& data, osmium::object_id_type start)
{
    std::unordered_set<osmium::object_id_type> visited;
    std::queue<osmium::object_id_type> q;
    q.push(start);
    visited.insert(start);
    while (!q.empty()) {
        auto cur = q.front(); q.pop();
        auto nit = data.nodes.find(cur);
        if (nit == data.nodes.end()) continue;
        for (auto way_id : nit->second.incident_ways) {
            auto wit = data.ways.find(way_id);
            if (wit == data.ways.end()) continue;
            for (auto nb : { wit->second.node1_id, wit->second.node2_id }) {
                if (nb != cur && !visited.count(nb)) {
                    visited.insert(nb);
                    q.push(nb);
                }
            }
        }
    }
    return visited.size();
}

// Vérifie un seul cache. Retourne true si fully connected (0 nœud isolé).
static bool check_cache(const std::string& path)
{
    if (!GeoBoxManager::cache_exists(path)) {
        std::cout << "  [SKIP] " << path << " — fichier introuvable\n";
        return true;  // pas une erreur de connectivité
    }

    GeoBox box = GeoBoxManager::load_geobox(path);
    if (!box.is_valid || box.data.nodes.empty()) {
        std::cout << "  [FAIL] " << path << " — GeoBox invalide ou vide\n";
        return false;
    }

    const size_t total = box.data.nodes.size();
    // Prend le premier nœud disponible comme point de départ BFS.
    const auto start = box.data.nodes.begin()->first;
    const size_t reachable = bfs_count(box.data, start);
    const size_t isolated  = total - reachable;

    // Extrait juste le nom de fichier pour l'affichage.
    const std::string name = path.substr(path.find_last_of("/\\") + 1);

    if (isolated == 0) {
        std::cout << "  [OK]   " << name
                  << "  nodes=" << total
                  << "  ways="  << box.data.ways.size()
                  << "  fully connected\n";
        return true;
    } else {
        std::cout << "  [FAIL] " << name
                  << "  nodes=" << total
                  << "  reachable=" << reachable
                  << "  isolated=" << isolated
                  << "  (" << (100.0 * isolated / total) << "% isolés)\n";
        return false;
    }
}

bool run_geobox_connectivity_test(const std::vector<std::string>& cache_paths,
                                  const std::string& label)
{
    std::cout << "\n=== GeoBox Connectivity — " << label << " ===\n";
    bool all_ok = true;
    for (const auto& p : cache_paths)
        if (!check_cache(p)) all_ok = false;

    if (all_ok) std::cout << "=== PASS — tous les caches sont connexes ===\n\n";
    else        std::cout << "=== FAIL — certains caches ont des nœuds isolés ===\n\n";
    return all_ok;
}