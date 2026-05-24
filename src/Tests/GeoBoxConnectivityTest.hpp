#ifndef GEOBOX_CONNECTIVITY_TEST_HPP
#define GEOBOX_CONNECTIVITY_TEST_HPP

#include <string>
#include <vector>

// Vérifie la connectivité de chaque cache GeoBox :
//   - Charge le cache JSON
//   - BFS non-dirigé depuis un nœud arbitraire
//   - Vérifie que tous les nœuds sont atteints (composante unique)
//
// Affiche un rapport par cache : nœuds total, nœuds atteints, nœuds isolés.
// Retourne true si TOUS les caches passent (0 nœud isolé).
//
// Usage :
//   run_geobox_connectivity_test({ path1, path2, ... }, label)
bool run_geobox_connectivity_test(const std::vector<std::string>& cache_paths,
                                  const std::string& label);

#endif // GEOBOX_CONNECTIVITY_TEST_HPP