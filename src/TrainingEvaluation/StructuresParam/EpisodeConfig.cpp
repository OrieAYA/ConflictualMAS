#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include <algorithm>

void ComparisonMetrics::on_task_completed(int latency_steps, float t_ratio) {
    ++tasks_completed;
    ++n_completed_acc_;
    latency_mean = (latency_mean * (n_completed_acc_ - 1) + latency_steps) / n_completed_acc_;
    avg_t_ratio  = (avg_t_ratio  * (n_completed_acc_ - 1) + t_ratio)       / n_completed_acc_;
    if (best_t_ratio == 0.f || t_ratio < best_t_ratio) best_t_ratio = t_ratio;
}

void ComparisonMetrics::on_task_refused() {
    ++tasks_appeared;
}

void ComparisonMetrics::on_offer(bool accepted, float reward) {
    ++tasks_appeared;
    ++n_offered_acc_;
    float a = accepted ? 1.f : 0.f;
    accept_rate  = (accept_rate  * (n_offered_acc_ - 1) + a)      / n_offered_acc_;
    reward_mean  = (reward_mean  * (n_offered_acc_ - 1) + reward)  / n_offered_acc_;
}

void ComparisonMetrics::finalise(int mean_active_agents, int steps_run) {
    total_steps = steps_run;
    if (tasks_appeared > 0)
        throughput_rate = static_cast<float>(tasks_completed) / tasks_appeared;
    refuse_rate = 1.f - accept_rate;
    if (mean_active_agents > 0)
        latency_per_agent = latency_mean / mean_active_agents;
}
