#include "EpisodeConfig.hpp"
#include <algorithm>

void ComparisonMetrics::accumulate_completion(int latency_steps, float quality) {
    ++tasks_in_window;
    latency_mean_steps = (latency_mean_steps * (tasks_in_window - 1) + latency_steps)
                         / tasks_in_window;
    avg_route_quality  = (avg_route_quality  * (tasks_in_window - 1) + quality)
                         / tasks_in_window;
    best_route_quality = (best_route_quality == 0.f)
                         ? quality
                         : std::min(best_route_quality, quality);
}

void ComparisonMetrics::accumulate_refusal() {
    ++tasks_total;
}

void ComparisonMetrics::accumulate_offer(bool accepted, float reward) {
    ++tasks_total;
    if (accepted) {
        policy_accept_rate  = (policy_accept_rate * (tasks_total - 1) + 1.f) / tasks_total;
        policy_reward_mean  = (policy_reward_mean * (tasks_total - 1) + reward) / tasks_total;
    } else {
        policy_accept_rate  = (policy_accept_rate * (tasks_total - 1)) / tasks_total;
    }
}

void ComparisonMetrics::finalise(int n_agents_active, int total_steps_run) {
    total_steps = total_steps_run;
    if (tasks_total > 0) {
        throughput_rate  = static_cast<float>(tasks_in_window) / tasks_total;
        tasks_refused    = 1.f - policy_accept_rate;
    }
    if (n_agents_active > 0)
        latency_per_agent = latency_mean_steps / n_agents_active;
}
