#include "TrainingLogger.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace fs = std::filesystem;

// ── Construction ─────────────────────────────────────────────────────────────

TrainingLogger::TrainingLogger(const std::string& output_dir, int seed) {
    fs::create_directories(output_dir);
    std::string path = output_dir + "/episodes_seed" + std::to_string(seed) + ".csv";
    file_.open(path);
    if (!file_.is_open())
        throw std::runtime_error("TrainingLogger: cannot open " + path);
    write_header();
}

TrainingLogger::~TrainingLogger() {
    if (file_.is_open()) file_.close();
}

// ── Header ───────────────────────────────────────────────────────────────────

void TrainingLogger::write_header() {
    file_ << "seed,global_episode,city,phase,policy_mode,"
          << "total_steps,n_agents_max,"
          << "tasks_appeared,tasks_completed,throughput_rate,"
          << "accept_rate,refuse_rate,"
          << "latency_mean,latency_per_agent,agent_utilisation,"
          << "mean_congestion,mean_trip_steps,"
          << "actor_loss,critic_loss,entropy,adv_std,n_experiences,"
          << "wallclock_ms\n";
    header_written_ = true;
}

// ── Row write ────────────────────────────────────────────────────────────────

void TrainingLogger::write_row(const EpisodeRecord& r) {
    file_ << r.seed             << ','
          << r.global_episode   << ','
          << r.city             << ','
          << r.phase            << ','
          << r.policy_mode      << ','
          << r.total_steps      << ','
          << r.n_agents_max     << ','
          << r.tasks_appeared   << ','
          << r.tasks_completed  << ','
          << std::fixed << std::setprecision(4)
          << r.throughput_rate  << ','
          << r.accept_rate      << ','
          << r.refuse_rate      << ','
          << r.latency_mean     << ','
          << r.latency_per_agent<< ','
          << r.agent_utilisation<< ','
          << r.mean_congestion  << ','
          << r.mean_trip_steps  << ','
          << r.actor_loss       << ','
          << r.critic_loss      << ','
          << r.entropy          << ','
          << r.adv_std          << ','
          << r.n_experiences    << ','
          << r.wallclock_ms     << '\n';
}

// ── Public interface ─────────────────────────────────────────────────────────

void TrainingLogger::push(const EpisodeRecord& r) {
    records_.push_back(r);
    write_row(r);
}

void TrainingLogger::flush() {
    file_.flush();
}

// ── Cross-seed summary ───────────────────────────────────────────────────────

void TrainingLogger::write_summary(const std::string& path,
                                   const std::vector<EpisodeRecord>& records,
                                   int seed) {
    bool exists = fs::exists(path);
    std::ofstream f(path, std::ios::app);
    if (!f.is_open()) return;

    if (!exists) {
        f << "seed,city,phase,policy_mode,"
          << "n_episodes,"
          << "throughput_mean,throughput_std,"
          << "accept_rate_mean,accept_rate_std,"
          << "latency_mean_mean,latency_mean_std,"
          << "utilisation_mean,utilisation_std,"
          << "mean_congestion_mean,mean_congestion_std,"
          << "mean_trip_steps_mean,mean_trip_steps_std,"
          << "actor_loss_mean,critic_loss_mean,entropy_mean\n";
    }

    // Group by (city, phase, policy_mode) and compute stats.
    struct Key { std::string city, phase, mode; };
    auto key_of = [](const EpisodeRecord& r) {
        return r.city + "|" + r.phase + "|" + r.policy_mode;
    };

    std::vector<std::string> seen_keys;
    for (const auto& r : records) {
        std::string k = key_of(r);
        if (std::find(seen_keys.begin(), seen_keys.end(), k) == seen_keys.end())
            seen_keys.push_back(k);
    }

    for (const auto& k : seen_keys) {
        std::vector<const EpisodeRecord*> group;
        for (const auto& r : records)
            if (key_of(r) == k) group.push_back(&r);
        if (group.empty()) continue;

        auto mean_of = [&](auto fn) {
            float s = 0.f;
            for (const auto* r : group) s += fn(*r);
            return s / group.size();
        };
        auto std_of = [&](auto fn, float mu) {
            float s = 0.f;
            for (const auto* r : group) { float d = fn(*r) - mu; s += d * d; }
            return std::sqrt(s / group.size());
        };

        float thr_m  = mean_of([](const EpisodeRecord& r){ return r.throughput_rate; });
        float acc_m  = mean_of([](const EpisodeRecord& r){ return r.accept_rate; });
        float lat_m  = mean_of([](const EpisodeRecord& r){ return r.latency_mean; });
        float uti_m  = mean_of([](const EpisodeRecord& r){ return r.agent_utilisation; });
        float cng_m  = mean_of([](const EpisodeRecord& r){ return r.mean_congestion; });
        float trp_m  = mean_of([](const EpisodeRecord& r){ return r.mean_trip_steps; });

        f << seed << ','
          << group[0]->city << ',' << group[0]->phase << ',' << group[0]->policy_mode << ','
          << group.size()   << ','
          << std::fixed << std::setprecision(4)
          << thr_m  << ',' << std_of([](const EpisodeRecord& r){ return r.throughput_rate; }, thr_m) << ','
          << acc_m  << ',' << std_of([](const EpisodeRecord& r){ return r.accept_rate; },    acc_m) << ','
          << lat_m  << ',' << std_of([](const EpisodeRecord& r){ return r.latency_mean; },   lat_m) << ','
          << uti_m  << ',' << std_of([](const EpisodeRecord& r){ return r.agent_utilisation; },uti_m) << ','
          << cng_m  << ',' << std_of([](const EpisodeRecord& r){ return r.mean_congestion; }, cng_m) << ','
          << trp_m  << ',' << std_of([](const EpisodeRecord& r){ return r.mean_trip_steps; }, trp_m) << ','
          << mean_of([](const EpisodeRecord& r){ return r.actor_loss; })  << ','
          << mean_of([](const EpisodeRecord& r){ return r.critic_loss; }) << ','
          << mean_of([](const EpisodeRecord& r){ return r.entropy; })     << '\n';
    }
}

// ── Helper ───────────────────────────────────────────────────────────────────

EpisodeRecord make_record(const RunResult& result,
                          int seed, int global_episode,
                          const std::string& city,
                          const std::string& phase,
                          const std::string& policy_mode_str,
                          int n_agents_max) {
    EpisodeRecord r;
    r.seed              = seed;
    r.global_episode    = global_episode;
    r.city              = city;
    r.phase             = phase;
    r.policy_mode       = policy_mode_str;
    r.n_agents_max      = n_agents_max;

    const auto& m = result.metrics;
    r.total_steps       = m.total_steps;
    r.tasks_appeared    = m.tasks_appeared;
    r.tasks_completed   = m.tasks_completed;
    r.throughput_rate   = m.throughput_rate;
    r.accept_rate       = m.accept_rate;
    r.refuse_rate       = m.refuse_rate;
    r.latency_mean      = m.latency_mean;
    r.latency_per_agent = m.latency_per_agent;
    r.agent_utilisation = m.agent_utilisation;
    r.mean_congestion   = m.mean_congestion;
    r.mean_trip_steps   = m.mean_trip_steps;

    const auto& ts = result.train_stats;
    r.actor_loss        = ts.actor_loss;
    r.critic_loss       = ts.critic_loss;
    r.entropy           = ts.entropy;
    r.adv_std           = ts.adv_std;
    r.n_experiences     = ts.n_exp;

    r.wallclock_ms      = result.wallclock_ms;
    return r;
}
