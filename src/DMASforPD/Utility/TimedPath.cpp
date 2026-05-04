#include "TimedPath.hpp"

void TimedPath::seal() {
    prefix_steps.resize(edge_steps.size() + 1, 0);
    for (std::size_t i = 0; i < edge_steps.size(); ++i)
        prefix_steps[i + 1] = prefix_steps[i] + edge_steps[i];
}

int TimedPath::rel_entry(std::size_t i) const {
    if (!prefix_steps.empty()) return prefix_steps[i];
    int acc = 0;
    for (std::size_t j = 0; j < i; ++j) acc += edge_steps[j];
    return acc;
}

int TimedPath::rel_exit(std::size_t i) const {
    if (!prefix_steps.empty()) return prefix_steps[i + 1];
    return rel_entry(i) + (i < edge_steps.size() ? edge_steps[i] : 0);
}
