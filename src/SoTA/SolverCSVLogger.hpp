#ifndef SOTA_SOLVER_CSV_LOGGER_HPP
#define SOTA_SOLVER_CSV_LOGGER_HPP

#include "SolverMetrics.hpp"
#include <fstream>
#include <string>

// ════════════════════════════════════════════════════════════════════════════
// SolverCSVLogger
// ════════════════════════════════════════════════════════════════════════════
//
// Writes SolverMetrics records to a CSV file, one row per (solver, city,
// scenario, episode). The schema mirrors the most useful columns of
// ComparisonMetrics (Training/TrainingLogger.cpp) so SoTA standalone runs
// and MAPPO/Hybrid runs can be merged for cross-method analysis.
//
// USAGE:
//   SolverCSVLogger log("results/sota_standalone/run.csv");
//   log.write_header();
//   for each (city, scenario, ep):
//     for each solver:
//       metrics = SolverRunner.run(solver);
//       log.write_row(metrics);
//   log.close();
//
// The file is opened in TRUNCATE mode by default — caller is responsible for
// not overwriting existing results. Pass append=true to extend an existing
// file (header is then NOT written; caller must have written it on creation).

class SolverCSVLogger {
public:
    SolverCSVLogger(const std::string& path, bool append = false);
    ~SolverCSVLogger();

    // Writes the CSV header. Idempotent — only writes if the file is empty.
    void write_header();

    // Writes one row from a SolverMetrics record. Fields not set by the
    // solver are written as zero.
    void write_row(const SolverMetrics& m);

    // Flushes and closes the underlying stream.
    void close();

    bool is_open() const { return out_.is_open(); }

private:
    std::ofstream out_;
    bool          header_written_ = false;
};

#endif // SOTA_SOLVER_CSV_LOGGER_HPP