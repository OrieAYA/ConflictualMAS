#include "SolverCSVLogger.hpp"
#include <iostream>

SolverCSVLogger::SolverCSVLogger(const std::string& path, bool append) {
    out_.open(path, append ? (std::ios::out | std::ios::app) : std::ios::out);
    if (!out_.is_open()) {
        std::cerr << "[SolverCSVLogger] failed to open " << path << "\n";
        return;
    }
    if (append) header_written_ = true;  // assume caller already has header
}

SolverCSVLogger::~SolverCSVLogger() {
    close();
}

void SolverCSVLogger::close() {
    if (out_.is_open()) {
        out_.flush();
        out_.close();
    }
}

void SolverCSVLogger::write_header() {
    if (header_written_ || !out_.is_open()) return;
    out_ <<
        // ── Identity ──
        "solver,city,scenario,episode,total_steps,n_agents,"
        // ── Throughput / counts ──
        "tasks_appeared,tasks_completed,tasks_refused,"
        "throughput_rate,accept_rate,"
        // ── Time ──
        "latency_mean,latency_per_agent,"
        "mean_wait_steps,mean_trip_steps,mean_road_pd_m,"
        // ── Routing ──
        "mean_extra_steps_per_task,delivery_route_efficiency,"
        "total_fleet_distance_m,"
        // ── Agents ──
        "agent_utilisation,agent_completed_max,agent_completed_min,"
        "agent_completed_gini,agent_completed_std,"
        // ── Network congestion ──
        "mean_congestion,congestion_variance,peak_load,n_ghost_active_mean,"
        // ── Congestion exposure ──
        "mean_congestion_at_decision,mean_bpr_along_route,"
        "time_lost_to_congestion,n_traversals_in_jam,route_congestion_exposure,"
        // ── Compute ──
        "wallclock_ms,n_allocation_calls,"
        "compute_time_per_task_ms,compute_time_per_decision_us,"
        // ── Solver bookkeeping ──
        "n_replans,n_collisions_avoided,"
        // ── Validity ──
        "capacity_violations,pairing_violations"
        << '\n';
    header_written_ = true;
}

void SolverCSVLogger::write_row(const SolverMetrics& m) {
    if (!out_.is_open()) return;
    if (!header_written_) write_header();

    out_ << m.solver_name << ','
         << m.city_label << ','
         << m.scenario_label << ','
         << m.episode << ','
         << m.total_steps << ','
         << m.n_active_agents << ','
         << m.tasks_appeared << ','
         << m.tasks_completed << ','
         << m.tasks_refused << ','
         << m.throughput_rate << ','
         << m.accept_rate << ','
         << m.latency_mean << ','
         << m.latency_per_agent << ','
         << m.mean_wait_steps << ','
         << m.mean_trip_steps << ','
         << m.mean_road_pd_m << ','
         << m.mean_extra_steps_per_task << ','
         << m.delivery_route_efficiency << ','
         << m.total_fleet_distance_m << ','
         << m.agent_utilisation << ','
         << m.agent_completed_max << ','
         << m.agent_completed_min << ','
         << m.agent_completed_gini << ','
         << m.agent_completed_std << ','
         << m.mean_congestion << ','
         << m.congestion_variance << ','
         << m.peak_load << ','
         << m.n_ghost_active_mean << ','
         << m.mean_congestion_at_decision << ','
         << m.mean_bpr_along_route << ','
         << m.time_lost_to_congestion << ','
         << m.n_traversals_in_jam << ','
         << m.route_congestion_exposure << ','
         << m.wallclock_ms << ','
         << m.n_allocation_calls << ','
         << m.compute_time_per_task_ms << ','
         << m.compute_time_per_decision_us << ','
         << m.n_replans << ','
         << m.n_collisions_avoided << ','
         << m.capacity_violations << ','
         << m.pairing_violations
         << '\n';
    out_.flush();
}