# ============================================================================
# analysis_paper.R
# ----------------------------------------------------------------------------
# Builds on analysis_eval.R + analysis_extended.R (UNCHANGED data layout and
# helper functions: load_eval, add_derived, plot_table, mat_by, trio_bar,
# rm_trio, heatmap_draw, heatmap_facet_panel, generalization_table, ...).
#
# Diagnostic on the real dataset (results/paper_eval, seed 42->45,
# RM 1.0->2.5, 5 cities x 2 sizes): the path-resolution logic in
# analysis_eval.R already matches this layout as-is (folder tag = "<City>_<Size>",
# captured generically by the "pol(\\d+)_rm([0-9.]+)_(.+)" regex). No structural
# change needed there. This file only ADDS the paper-specific tables/figures
# agreed on with the user.
#
# 3 core metrics (confirmed):
#   Throughput = throughput_rate
#   Latency    = latency_mean (priority) ; mean_trip_steps = pickup->delivery,
#                secondary / not general -> only in Table S3 (planning detail)
#   Congestion = mean_bpr_along_route (mean BPR along executed routes,
#                averaged across all traversed edges -- confirmed correct)
#
# MAIN PAPER (6 methods: MAPPO, Hybrid, RMCA, TokenPassing, CA, HAPC), 4
# elements, only 1 visual:
#   Table 1  - overall performance                       (table)
#   Table 2  - generalization + all cities + Small->Medium delta%  (table)
#   Figure 1 - scalability vs RM load (1.0 -> 2.5)        (THE visual)
#   Figure 2 - scenario-facet heatmap (throughput)        (visual, element 4)
#
# SUPPLEMENTARY MATERIALS (4 RL policies: MAPPO, Hybrid, IPPO, MAPPER --
# confirmed: same training protocol, same pipeline, same episodes, same seed):
#   Table S1  - overall performance
#   Table S2  - generalization + all cities + Small->Medium delta%
#   Figure S2 - trained vs unseen, 3-metric grouped bars
#   Figure S3 - scalability vs RM load
#   Figure S4 - scenario-facet heatmap
#   Table S3  - planning-efficiency detail (secondary/pickup->delivery latency,
#               explicitly "not general" -> supplementary only)
#
# All main-paper tables/figures are computed at RM = 1.0 (the training-protocol
# load), so the geographic-generalization effect is never confounded with the
# load-scaling effect studied separately in Figure 1/S3.
#
# USAGE:
#   source("analysis_eval.R"); source("analysis_extended.R"); source("analysis_paper.R")
#   run_paper_report()   # writes everything under EVAL_ROOT/plots/paper/
# ============================================================================

# ---------------------------------------------------------------------------
# Safety net: this file assumes analysis_eval.R and analysis_extended.R were
# already source()'d in that order (that's where load_eval, add_derived,
# METHOD_ORDER, PROPOSED_METHODS, plot_table, rm_trio, heatmap_facet_panel,
# trio_bar, overview_table, pick_metric, EVAL_ROOT, PLOT_DIR are defined).
# "object 'PROPOSED_METHODS' not found" means this file was run/sourced on
# its own, or before the other two. This block tries to auto-source them
# from the current working directory; if that fails, it stops with a clear
# message instead of a cryptic downstream error.
if (!exists("PROPOSED_METHODS") || !exists("plot_table")) {
  ok <- tryCatch({
    if (!exists("load_eval", mode = "function")) source("analysis_eval.R")
    if (!exists("PROPOSED_METHODS")) source("analysis_extended.R")
    TRUE
  }, error = function(e) FALSE)
  if (!ok || !exists("PROPOSED_METHODS") || !exists("plot_table")) {
    stop("analysis_eval.R / analysis_extended.R not found or not loaded.\n",
         "Fix: setwd() to the folder containing all three .R files, then run:\n",
         '  source("analysis_eval.R"); source("analysis_extended.R"); source("analysis_paper.R")\n',
         "  run_paper_report()")
  }
}

REF_RM <- 1.0

MAIN_METHODS <- c("MAPPO", "Hybrid", "RMCA", "TokenPassing", "CA", "HAPC")
SUPP_METHODS <- PROPOSED_METHODS   # already = intersect(c("MAPPO","IPPO","MAPPER","Hybrid"), METHOD_ORDER)
TRAINED_MAIN <- c("MAPPO", "Hybrid")   # trained-protocol methods present in the main-paper set

ALL_CITIES_ORDER <- c("Tokyo", "LosAngeles", "Paris", "Kyoto", "NewYork")
TRAIN_CITIES_DISP <- c("Tokyo", "LosAngeles", "Paris")
GEN_CITIES_DISP   <- c("Kyoto", "NewYork")

# ============================================================================
# TABLE 1 / S1 -- overall performance at RM = REF_RM
# (thin wrapper around perf_table(), restricted to a method set and RM level)
# ============================================================================
overall_table_rm <- function(df, methods, rm = REF_RM) {
  sub <- df[df$rm == rm & df$method %in% methods, ]
  sub$method <- factor(sub$method, levels = intersect(METHOD_ORDER, methods))
  overview_table(sub)
}

# ============================================================================
# TABLE 2 / S2 -- generalization (Trained vs Unseen) + full per-city
# breakdown + Small->Medium delta (%), on throughput_rate, at RM = REF_RM
# ============================================================================
generalization_table_full <- function(df, methods, trained_methods,
                                       rm = REF_RM, metric = "throughput_rate") {
  sub <- df[df$rm == rm & df$method %in% methods, ]
  sub$method <- droplevels(factor(sub$method, levels = intersect(METHOD_ORDER, methods)))
  meth <- levels(sub$method)

  by_city <- tapply(sub[[metric]], list(sub$town, sub$method), mean, na.rm = TRUE)
  by_city <- by_city[intersect(ALL_CITIES_ORDER, rownames(by_city)), meth, drop = FALSE]

  tr <- colMeans(by_city[TRAIN_CITIES_DISP, , drop = FALSE], na.rm = TRUE)
  un <- colMeans(by_city[GEN_CITIES_DISP, , drop = FALSE], na.rm = TRUE)

  small  <- tapply(sub[[metric]][sub$size == "Small"],  droplevels(sub$method[sub$size == "Small"]),  mean, na.rm = TRUE)[meth]
  medium <- tapply(sub[[metric]][sub$size == "Medium"], droplevels(sub$method[sub$size == "Medium"]), mean, na.rm = TRUE)[meth]

  r <- data.frame(Method = paste0(meth, ifelse(meth %in% trained_methods, " *", "")),
                   row.names = NULL, stringsAsFactors = FALSE)
  for (city in intersect(ALL_CITIES_ORDER, rownames(by_city)))
    r[[city]] <- round(by_city[city, ], 3)
  r$Trained_avg      <- round(tr, 3)
  r$Unseen_avg       <- round(un, 3)
  r$Delta_Unseen_pct <- round(100 * (un - tr) / tr, 1)
  r$Delta_SmallToMedium_pct <- round(100 * (medium - small) / small, 1)
  r[order(-r$Trained_avg), ]
}

# ============================================================================
# TABLE S3 -- planning-efficiency detail (secondary latency angle: pickup ->
# delivery). Explicitly flagged as "not general" -> supplementary only.
# ============================================================================
planning_efficiency_table <- function(df, methods = SUPP_METHODS, rm = REF_RM) {
  sub <- df[df$rm == rm & df$method %in% methods, ]
  sub$method <- droplevels(factor(sub$method, levels = intersect(METHOD_ORDER, methods)))
  s <- function(metric, f) tapply(sub[[metric]], sub$method, f, na.rm = TRUE)
  r <- data.frame(
    Method                 = levels(sub$method),
    PickupToDelivery_steps = round(s("mean_trip_steps", mean), 0),
    WaitBeforePickup_steps = round(s("mean_wait_steps", mean), 0),
    DetourStepsPerTask     = round(s("mean_extra_steps_per_task", mean), 2),
    RouteEfficiency        = round(s("delivery_route_efficiency", mean), 3),
    row.names = NULL, stringsAsFactors = FALSE)
  r[order(r$PickupToDelivery_steps), ]
}

# ============================================================================
# FIGURE S2 -- trained vs unseen, 3-metric grouped bars (companion to Table S2)
# ============================================================================
generalization_trio <- function(df, methods = SUPP_METHODS, rm = REF_RM, file = NULL) {
  sub <- df[df$rm == rm & df$method %in% methods & !is.na(df$city_group), ]
  trio_bar(sub, by = "city_group", methods = methods, file = file,
           title = "Trained (Tokyo/LA/Paris) vs Unseen (Kyoto/NewYork) cities, RM = 1.0")
}

# ============================================================================
# ORCHESTRATOR
# ============================================================================
run_paper_report <- function(root = EVAL_ROOT, dir = file.path(PLOT_DIR, "paper")) {
  df <- load_eval(root)
  df <- add_derived(df)
  dm <- file.path(dir, "main"); dir.create(dm, recursive = TRUE, showWarnings = FALSE)
  ds <- file.path(dir, "supplementary"); dir.create(ds, recursive = TRUE, showWarnings = FALSE)

  ## -- MAIN PAPER ------------------------------------------------------------
  plot_table(overall_table_rm(df, MAIN_METHODS), title = "Table 1 -- Overall performance (RM = 1.0)",
             file = file.path(dm, "Table1_overall_performance.png"))

  plot_table(generalization_table_full(df, MAIN_METHODS, TRAINED_MAIN),
             title = "Table 2 -- Generalization & city-scale sensitivity (RM = 1.0)",
             file = file.path(dm, "Table2_generalization.png"))

  rm_trio(df[df$method %in% MAIN_METHODS, ],
          file = file.path(dm, "Figure1_scale_RM.png"),
          title = "Figure 1 -- Scalability under increasing event load (RM = 1.0 -> 2.5)")

  heatmap_facet_panel(df[df$rm == REF_RM & df$method %in% MAIN_METHODS, ],
                       pick_metric(df, "throughput_rate"),
                       c("task_profile", "cong_profile", "agent_multiplier"),
                       file = file.path(dm, "Figure2_scenario_heatmap.png"),
                       title = "Figure 2 -- Throughput across scenario facets (RM = 1.0)")

  ## -- SUPPLEMENTARY MATERIALS ------------------------------------------------
  plot_table(overall_table_rm(df, SUPP_METHODS), title = "Table S1 -- Overall performance, RL policies (RM = 1.0)",
             file = file.path(ds, "TableS1_overall_performance_policies.png"))

  plot_table(generalization_table_full(df, SUPP_METHODS, SUPP_METHODS),
             title = "Table S2 -- Generalization & city-scale sensitivity, RL policies (RM = 1.0)",
             file = file.path(ds, "TableS2_generalization_policies.png"))

  generalization_trio(df, file = file.path(ds, "FigureS2_generalization_policies.png"))

  rm_trio(df[df$method %in% SUPP_METHODS, ],
          file = file.path(ds, "FigureS3_scale_RM_policies.png"),
          title = "Figure S3 -- Scalability under increasing event load, RL policies (RM = 1.0 -> 2.5)")

  heatmap_facet_panel(df[df$rm == REF_RM & df$method %in% SUPP_METHODS, ],
                       pick_metric(df, "throughput_rate"),
                       c("task_profile", "cong_profile", "agent_multiplier"),
                       file = file.path(ds, "FigureS4_scenario_heatmap_policies.png"),
                       title = "Figure S4 -- Throughput across scenario facets, RL policies (RM = 1.0)")

  plot_table(planning_efficiency_table(df), title = "Table S3 -- Planning-efficiency detail (RM = 1.0)",
             file = file.path(ds, "TableS3_planning_efficiency.png"))

  message("\n=== Paper report generated -> ", normalizePath(dir, winslash = "/"), " ===")
  invisible(df)
}
run_paper_report()