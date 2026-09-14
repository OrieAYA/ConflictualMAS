# ============================================================================
# analysis_extended.R  (v3 -- English output, readable metric names)
# ----------------------------------------------------------------------------
# Builds on analysis_eval.R (UNCHANGED data layout: results/paper_eval/
# pol{P}_rm{RM}_{grp}/episodes_seed*.csv + sota_standalone/sota_seed*.csv)
# and adds the analysis figures/tables discussed for the SIGSPATIAL'26 paper.
#
# 3 base metrics used everywhere below:
#   Throughput        -> throughput_rate      ("completion rate")
#   Latency             -> latency_mean         ("avg. steps from pickup to delivery")
#   Congestion           -> mean_bpr_along_route (repli : mean_congestion)
#
# 01_core/                 (UNCHANGED, does not count toward the 10-plot cap)
#   3 throughput plots + comparison table + evaluation progress
#
# 8 plots (cap = 10):
#   02_overview/                P1 multi-criteria overview, all 8 methods
#   03_scale_rm/                P2 load scaling (RM 1.0 -> 2.5)
#   04_scenario_comparison/     P3 by task profile
#                                P4 by congestion profile
#                                P5 by agent multiplier (AM)
#   05_geo_generalization/      P6 training vs unseen cities
#   06_scenario_heatmap/        P7 3 small heatmaps (one per scenario facet)
#   07_geo_scale/               P8 scale sensitivity Small -> Medium (delta view)
#
# 5 tables (cap = 5), in 08_tables/:
#   T1 overall comparison
#   T2 stability across repeats + compute cost
#   T3 geospatial scale (Small vs Medium, incl. deltas)
#   T4 generalization (training vs unseen cities, proposed methods)
#   T5 winning method per scenario facet
#
# 09_notes/ : auto-generated text summaries (bonus, not counted)
#
# USAGE ON THE EXPERIMENT MACHINE:
#   Rscript analysis_extended.R
#   (interactive: source("analysis_extended.R"); run_full_report())
# ============================================================================

## -- Load the base script (analysis_eval.R) ---------------------------------
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0 || !nzchar(a)) b else a

if (!exists("load_eval")) {
  .args <- commandArgs(trailingOnly = FALSE)
  .fa   <- grep("^--file=", .args, value = TRUE)
  .self_dir_rscript <- if (length(.fa)) dirname(sub("^--file=", "", .fa[1])) else NA_character_
  
  .self_dir_source <- tryCatch({
    ofile <- sys.frame(1)$ofile
    if (!is.null(ofile)) dirname(normalizePath(ofile, winslash = "/", mustWork = FALSE)) else NA_character_
  }, error = function(e) NA_character_)
  
  .root_guess <- NA_character_
  d <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  for (i in 1:8) {
    if (file.exists(file.path(d, "analysis_eval.R"))) { .root_guess <- d; break }
    p <- dirname(d); if (identical(p, d)) break
    d <- p
  }
  
  cands <- c("analysis_eval.R",
             file.path("scripts", "analysis_eval.R"),
             file.path(.self_dir_rscript, "analysis_eval.R"),
             file.path(.self_dir_source,  "analysis_eval.R"),
             file.path(.root_guess,       "analysis_eval.R"),
             "C:/ConflictualMAS/analysis_eval.R")
  cands <- cands[!is.na(cands) & nzchar(cands)]
  
  ok <- FALSE
  for (cand in cands) {
    if (file.exists(cand)) { message("sourcing: ", normalizePath(cand, winslash = "/")); source(cand); ok <- TRUE; break }
  }
  if (!ok) stop("analysis_eval.R not found (tried: ", paste(cands, collapse = " | "),
                "). Run this script from the folder containing analysis_eval.R, ",
                "or call setwd(\"C:/ConflictualMAS\") before sourcing it.")
}

# ============================================================================
# CONFIG: base metrics, direction (1 = higher is better, -1 = lower is better)
# and human-readable labels used in every plot/table.
# ============================================================================
METRIC_DIR <- c(
  throughput_rate            =  1,
  tasks_completed             =  1,
  agent_utilisation           =  1,
  delivery_route_efficiency   =  1,
  latency_mean                = -1,
  latency_per_agent           = -1,
  mean_wait_steps              = -1,
  mean_trip_steps              = -1,
  mean_extra_steps_per_task    = -1,
  distance_per_task            = -1,
  total_fleet_distance_m       = -1,
  mean_congestion               = -1,
  mean_bpr_along_route          = -1,
  route_congestion_exposure     = -1,
  n_traversals_in_jam           = -1,
  agent_completed_gini           = -1,
  compute_time_per_task_ms       = -1,
  wallclock_ms                    = -1
)
metric_dir <- function(m) { d <- METRIC_DIR[m]; if (is.na(d)) 1 else d }

METRIC_LABEL <- c(
  throughput_rate           = "Task completion rate",
  latency_mean               = "Avg. latency (steps)",
  latency_per_agent          = "Avg. latency per agent (steps)",
  mean_wait_steps              = "Avg. wait time (steps)",
  mean_bpr_along_route          = "Route congestion (BPR factor)",
  mean_congestion                = "Network congestion",
  distance_per_task                = "Distance per task (m)",
  mean_extra_steps_per_task         = "Extra steps per task (detour)",
  compute_time_per_task_ms           = "Compute time per task (ms)",
  agent_completed_gini                 = "Fairness (Gini of completions)"
)
metric_label <- function(m) { l <- METRIC_LABEL[m]; if (is.na(l)) m else unname(l) }

FACET_LABEL <- c(
  task_profile     = "Task profile",
  cong_profile     = "Congestion profile",
  agent_multiplier = "Agent multiplier (AM)",
  city_group       = "City type",
  size             = "City scale"
)
facet_label <- function(f) { l <- FACET_LABEL[f]; if (is.na(l)) f else unname(l) }

# short column codes used in tables (kept short so headers fit), always
# accompanied by a plain-English caption in the table title.
SHORT <- c(throughput_rate = "Compl", latency_mean = "Delay",
           mean_bpr_along_route = "Cong", mean_congestion = "Cong")
short_lbl <- function(m) { l <- SHORT[m]; if (is.na(l)) m else unname(l) }

PROPOSED_METHODS <- intersect(c("MAPPO", "IPPO", "MAPPER", "Hybrid"), METHOD_ORDER)

pick_metric <- function(df, candidates) {
  for (m in candidates) if (m %in% names(df) && any(!is.na(df[[m]]))) return(m)
  candidates[1]
}
base_metrics <- function(df)
  c(Throughput = pick_metric(df, "throughput_rate"),
    `Latency` = pick_metric(df, "latency_mean"),
    `Congestion` = pick_metric(df, c("mean_bpr_along_route", "mean_congestion")))

# ============================================================================
# SCENARIO FACETS -- rebuilt from the composite `scenario` field
# (e.g. "task_normal/cong_shock/agents_high") + city classification.
# ============================================================================
extract_facet <- function(scenario, prefix) {
  parts <- strsplit(as.character(scenario), "/", fixed = TRUE)
  vapply(parts, function(p) {
    hit <- grep(paste0("^", prefix), p, value = TRUE)
    if (length(hit)) hit[1] else NA_character_
  }, character(1))
}

# agent multiplier: agents_low/mid/high -> paper's AM grid {0.7, 1.0, 2.5}
# (falls back to a literal number if the label already carries one, e.g. agents_1.5)
AM_MAP <- c(agents_low = "0.7", agents_mid = "1.0", agents_high = "2.5")
normalize_am <- function(x) {
  out <- unname(ifelse(x %in% names(AM_MAP), AM_MAP[x], NA_character_))
  num <- regmatches(x, regexpr("[0-9]+\\.?[0-9]*$", x))
  miss <- is.na(out) & nzchar(num)
  out[miss] <- num[miss]
  out
}

# training cities vs unseen (generalization) cities -- paper, section 5.6
TRAIN_CITIES <- c("tokyo", "losangeles", "la", "paris")
GEN_CITIES   <- c("kyoto", "newyork", "ny")
classify_city_group <- function(town) {
  t <- tolower(gsub("[ _-]", "", town))
  ifelse(t %in% TRAIN_CITIES, "Training", ifelse(t %in% GEN_CITIES, "Generalization", NA_character_))
}

disp_suffix <- function(x, prefix) tools::toTitleCase(gsub("_", " ", sub(paste0("^", prefix), "", x)))
order_normal_first <- function(x) {
  lv <- sort(unique(x[!is.na(x)]))
  if ("Normal" %in% lv) lv <- c("Normal", setdiff(lv, "Normal"))
  factor(x, levels = lv)
}

add_derived <- function(df) {
  df$distance_per_task <- ifelse(!is.na(df$tasks_completed) & df$tasks_completed > 0,
                                 df$total_fleet_distance_m / df$tasks_completed, NA_real_)
  
  df$task_profile <- order_normal_first(disp_suffix(extract_facet(df$scenario, "task_"), "task_"))
  df$cong_profile <- order_normal_first(disp_suffix(extract_facet(df$scenario, "cong_"), "cong_"))
  
  am_num <- normalize_am(extract_facet(df$scenario, "agents_"))
  df$agent_multiplier <- factor(am_num, levels = intersect(c("0.7", "1.0", "2.5"), unique(am_num)))
  
  df$city_group <- classify_city_group(df$town)
  df
}

# ============================================================================
# GENERIC PLOT HELPERS
# ============================================================================

# -- 3-panel plot (Throughput / Latency / Congestion), bars --
# by = NULL      -> one bar per method
# by = <column>  -> bars grouped by method, one color per level of `by`
#                    (also prints a diagnostic if a bar has <2 samples, since
#                    sd() is then undefined and no error whisker is drawn --
#                    this reflects incomplete data, not a rendering bug)
trio_bar <- function(df, by = NULL, methods = NULL, file = NULL, title = "",
                     metrics = NULL, err = TRUE) {
  if (is.null(metrics)) metrics <- base_metrics(df)
  sub <- df
  if (!is.null(methods)) sub <- sub[sub$method %in% methods, ]
  if (!is.null(by)) { sub <- sub[!is.na(sub[[by]]), ]; sub[[by]] <- droplevels(factor(sub[[by]])) }
  sub$method <- droplevels(factor(sub$method))
  if (!nrow(sub)) { message("trio_bar: no data (methods/", by, ")"); return(invisible(NULL)) }
  
  to_file <- !is.null(file)
  if (to_file) {
    dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    png(file, width = 560 * length(metrics), height = 720, res = 120, pointsize = 16)
  }
  op <- par(mfrow = c(1, length(metrics)), oma = c(0, 0, 3.6, 0), mar = c(6.5, 5.4, 3.8, 1),
            cex.axis = 1.05, cex.lab = 1.3, cex.main = 1.55)
  grp_cols <- c("#1b6ca8", "#d1495b", "#2e9e5b", "#e08a1e")
  for (nm in names(metrics)) {
    met <- metrics[[nm]]
    if (!met %in% names(sub) || !any(!is.na(sub[[met]]))) { plot.new(); title(main = paste(nm, "(n/a)")); next }
    dirn <- metric_dir(met)
    if (is.null(by)) {
      mu  <- tapply(sub[[met]], sub$method, mean, na.rm = TRUE)
      sdv <- tapply(sub[[met]], sub$method, sd,   na.rm = TRUE)
      o   <- order(mu, decreasing = (dirn == 1))
      cols <- METHOD_COL[names(mu)[o]]
      bp <- barplot(mu[o], col = cols, border = NA, las = 2, cex.names = .95,
                    ylim = c(0, max(mu[o] + sdv[o], na.rm = TRUE) * 1.15),
                    main = nm, ylab = metric_label(met))
      if (err) segments(bp, mu[o] - sdv[o], bp, mu[o] + sdv[o])
    } else {
      mu  <- tapply(sub[[met]], list(sub[[by]], sub$method), mean, na.rm = TRUE)
      sdv <- tapply(sub[[met]], list(sub[[by]], sub$method), sd,   na.rm = TRUE)
      nn  <- tapply(sub[[met]], list(sub[[by]], sub$method), function(v) sum(!is.na(v)))
      mu  <- mu[, colSums(!is.na(mu)) > 0, drop = FALSE]
      sdv <- sdv[rownames(mu), colnames(mu), drop = FALSE]
      nn  <- nn[rownames(mu), colnames(mu), drop = FALSE]
      sdv[nn < 2] <- NA  # sd needs >=2 samples; NA here means "not enough repeats yet"
      cols <- grp_cols[seq_len(nrow(mu))]
      bp <- barplot(mu, beside = TRUE, col = cols, border = NA, las = 2, cex.names = .95,
                    ylim = c(0, max(mu + sdv, na.rm = TRUE) * 1.18),
                    main = nm, ylab = metric_label(met),
                    legend.text = rownames(mu),
                    args.legend = list(x = "top", bty = "n", ncol = nrow(mu), cex = .92))
      if (err) segments(bp, mu - sdv, bp, mu + sdv)
      thin <- colnames(mu)[colSums(nn < 2, na.rm = TRUE) > 0]
      if (length(thin))
        message("note [", nm, "]: no error bar for ", paste(thin, collapse = ", "),
                " (fewer than 2 completed repeats for at least one group yet)")
    }
  }
  mtext(title, outer = TRUE, font = 2, cex = 1.4)
  par(op)
  if (to_file) { dev.off(); message("plot -> ", normalizePath(file, winslash = "/")) }
}

# -- single-metric cross-scenario comparison (grouped bars: method x facet level) --
facet_bar <- function(df, metric, by, file = NULL, title = NULL, err = TRUE) {
  sub <- df[!is.na(df[[by]]), ]
  sub[[by]] <- droplevels(factor(sub[[by]]))
  sub$method <- droplevels(factor(sub$method))
  if (!nrow(sub) || nlevels(sub[[by]]) < 1) { message("facet_bar: no levels for ", by); return(invisible(NULL)) }
  mu  <- tapply(sub[[metric]], list(sub$method, sub[[by]]), mean, na.rm = TRUE)
  sdv <- tapply(sub[[metric]], list(sub$method, sub[[by]]), sd,   na.rm = TRUE)
  mu  <- mu[rowSums(!is.na(mu)) > 0, , drop = FALSE]
  sdv <- sdv[rownames(mu), colnames(mu), drop = FALSE]
  cols <- METHOD_COL[rownames(mu)]
  
  to_file <- !is.null(file)
  if (to_file) { dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    png(file, width = 1500, height = 900, res = 120, pointsize = 16) }
  op <- par(mar = c(5.4, 5.8, 3.8, 1), cex.axis = 1.05, cex.lab = 1.3, cex.main = 1.55)
  bp <- barplot(mu, beside = TRUE, col = cols, border = NA,
                ylim = c(0, max(mu + sdv, na.rm = TRUE) * 1.18),
                ylab = metric_label(metric), xlab = facet_label(by),
                main = if (is.null(title)) paste(metric_label(metric), "by", facet_label(by)) else title,
                cex.names = 1.05,
                legend.text = rownames(mu),
                args.legend = list(x = "top", bty = "n", ncol = 4, cex = 1))
  if (err) segments(bp, mu - sdv, bp, mu + sdv)
  par(op)
  if (to_file) { dev.off(); message("plot -> ", normalizePath(file, winslash = "/")) }
  invisible(mu)
}

# -- P2: 3 panels (Throughput/Latency/Congestion) vs RM ----------
rm_trio <- function(df, file = NULL, title = "Load scaling (RM = event ratio multiplier)") {
  metrics <- base_metrics(df)
  to_file <- !is.null(file)
  if (to_file) { dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    png(file, width = 800 * length(metrics), height = 820, res = 120, pointsize = 17) }
  op <- par(mfrow = c(1, length(metrics)), oma = c(0, 0, 3.8, 0), mar = c(4.8, 6, 3.8, 1),
            cex.axis = 1.15, cex.lab = 1.4, cex.main = 1.65)
  for (nm in names(metrics)) {
    met <- metrics[[nm]]
    m <- mat_by(df, met, "rm")
    m <- m[, colSums(!is.na(m)) > 0, drop = FALSE]
    if (!ncol(m)) { plot.new(); title(main = paste(nm, "(n/a)")); next }
    x <- as.numeric(colnames(m))
    cols <- METHOD_COL[rownames(m)]
    if (length(x) < 2) {
      o <- order(m[, 1], decreasing = (metric_dir(met) == 1))
      barplot(m[o, 1], col = cols[o], las = 2, cex.names = 1.05, main = nm, ylab = metric_label(met))
    } else {
      yr <- range(m, na.rm = TRUE)
      leg_n <- nrow(m); leg_ncol <- if (leg_n <= 3) leg_n else 2
      leg_rows <- ceiling(leg_n / leg_ncol)
      matplot(x, t(m), type = "b", pch = 16, cex = 1.15, lty = 1, lwd = 4, col = cols, xaxt = "n",
              xlab = "Ratio Multiplier (RM)", ylab = metric_label(met), main = nm,
              ylim = yr + c(-.16 * leg_rows - .12, .08) * max(diff(yr), 1e-9))
      axis(1, at = x, labels = format(x, nsmall = 1))
      grid(col = "grey85")
      if (nm == names(metrics)[1])
        legend("bottom", rownames(m), col = cols, lty = 1, lwd = 4, pch = 16, bty = "n",
               cex = 1, ncol = leg_ncol)
    }
  }
  mtext(title, outer = TRUE, font = 2, cex = 1.5)
  par(op)
  if (to_file) { dev.off(); message("plot -> ", normalizePath(file, winslash = "/")) }
}

# -- P7: small readable heatmaps, one per scenario facet (method x 3 levels) --
heatmap_draw <- function(df, metric, by, main = NULL, show_labels = TRUE) {
  m <- mat_by(df, metric, by)
  if (nrow(m) < 2 || ncol(m) < 1) { plot.new(); if (!is.null(main)) title(main = paste(main, "(n/a)")); return(invisible(NULL)) }
  dirn <- metric_dir(metric)
  z <- apply(m, 2, function(col) {
    r <- range(col, na.rm = TRUE)
    if (diff(r) == 0 || all(is.na(r))) return(rep(0.5, length(col)))
    v <- (col - r[1]) / diff(r)
    if (dirn == -1) v <- 1 - v
    v
  })
  dimnames(z) <- dimnames(m)
  nR <- nrow(z); nC <- ncol(z)
  image(x = seq_len(nC), y = seq_len(nR), z = t(z), axes = FALSE, xlab = "", ylab = "",
        col = colorRampPalette(c("#c0392b", "#f4d35e", "#2e9e5b"))(100),
        main = main, cex.main = 1.5, asp = 1)
  axis(1, at = seq_len(nC), labels = colnames(z), las = 1, cex.axis = 1.15)
  if (show_labels) axis(2, at = seq_len(nR), labels = rownames(z), las = 1, cex.axis = 1.15)
  for (i in seq_len(nR)) for (j in seq_len(nC))
    if (!is.na(m[i, j]))
      text(j, i, format(round(m[i, j], 2), nsmall = 2), cex = 1.3, font = 2,
           col = ifelse(z[i, j] > .5, "grey15", "white"))
  box()
  invisible(z)
}

# -- shared method-name column (drawn once, on the far left; centred, no header) --
heatmap_row_labels <- function(meth) {
  nR <- length(meth)
  plot.new(); plot.window(c(0, 1), c(nR + .5, .5))
  text(0.5, seq_len(nR), meth, adj = c(0.5, 0.5), font = 2, cex = 1.3, xpd = NA)
}

heatmap_facet_panel <- function(df, metric, facets, file = NULL,
                                title = "Cross-scenario comparison  (green = best method, red = worst)") {
  m_ref <- mat_by(df, metric, facets[1])
  meth  <- rownames(m_ref)
  nR    <- length(meth)
  nC    <- max(vapply(facets, function(fa) ncol(mat_by(df, metric, fa)), integer(1)))

  cell_px <- 185                                   # one heatmap cell, in px -- kept square via asp=1
  mar_b <- 5.4; mar_l <- 1; mar_t <- 3.8; mar_r <- 0.3
  line_px <- 17 / 72 * 120                          # ~px per margin "line" at pointsize 17, res 120
  panel_w <- round(nC * cell_px + (mar_l + mar_r) * line_px)
  panel_h <- round(nR * cell_px + (mar_t + mar_b) * line_px)
  label_w <- 300
  oma_top_lines <- 5.6
  oma_top <- round(oma_top_lines * line_px)

  to_file <- !is.null(file)
  if (to_file) { dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    png(file, width = label_w + panel_w * length(facets), height = panel_h + oma_top, res = 120, pointsize = 17) }
  op <- par(oma = c(0, 0, oma_top_lines, 0), mar = c(mar_b, mar_l, mar_t, mar_r))
  layout(matrix(seq_len(length(facets) + 1), nrow = 1),
         widths = c(label_w, rep(panel_w, length(facets))))
  heatmap_row_labels(meth)
  for (fa in facets) heatmap_draw(df, metric, fa, main = facet_label(fa), show_labels = FALSE)
  mtext(title, outer = TRUE, font = 2, cex = 1.5, line = 2.6)
  mtext(paste0("(", metric_label(metric), ")"), outer = TRUE, font = 3, cex = 1.15, line = 0.9)
  par(op)
  if (to_file) { dev.off(); message("plot -> ", normalizePath(file, winslash = "/")) }
}

# -- P8: scale sensitivity as a %-change view (clearer than raw side-by-side bars) --
delta_trio <- function(df, by, ref, target, methods = NULL, file = NULL, title = "") {
  metrics <- base_metrics(df)
  sub <- df
  if (!is.null(methods)) sub <- sub[sub$method %in% methods, ]
  sub <- sub[!is.na(sub[[by]]) & as.character(sub[[by]]) %in% c(ref, target), ]
  sub$method <- droplevels(factor(sub$method))
  if (!nrow(sub)) { message("delta_trio: no data"); return(invisible(NULL)) }
  
  to_file <- !is.null(file)
  if (to_file) { dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    png(file, width = 530 * length(metrics), height = 700, res = 120, pointsize = 16) }
  op <- par(mfrow = c(1, length(metrics)), oma = c(0, 0, 3.8, 0), mar = c(4.6, 9.5, 3.8, 1),
            cex.axis = 1.05, cex.lab = 1.3, cex.main = 1.55)
  for (nm in names(metrics)) {
    met <- metrics[[nm]]
    m <- tapply(sub[[met]], list(sub$method, as.character(sub[[by]])), mean, na.rm = TRUE)
    if (!all(c(ref, target) %in% colnames(m))) { plot.new(); title(main = paste(nm, "(n/a)")); next }
    delta <- 100 * (m[, target] - m[, ref]) / abs(m[, ref])
    delta <- sort(delta[!is.na(delta)])
    dirn <- metric_dir(met)
    good <- (dirn == 1 & delta > 0) | (dirn == -1 & delta < 0)
    cols <- ifelse(good, "#2e9e5b", "#c0392b")
    bp <- barplot(delta, horiz = TRUE, las = 1, col = cols, border = NA, cex.names = .95,
                  xlab = paste0("% change (", ref, " -> ", target, ")"), main = nm)
    abline(v = 0, col = "grey40")
    text(delta, bp, labels = paste0(round(delta, 1), "%"), pos = ifelse(delta >= 0, 4, 2), cex = .88)
  }
  mtext(title, outer = TRUE, font = 2, cex = 1.4)
  par(op)
  if (to_file) { dev.off(); message("plot -> ", normalizePath(file, winslash = "/")) }
}

# ============================================================================
# TABLES (T1-T5)
# ============================================================================

# T1: overall comparison (reuses perf_table, trimmed + relabeled columns)
overview_table <- function(df) {
  p <- perf_table(df)
  p <- p[, intersect(c("Method", "Thr", "Lat", "BPR", "N"), names(p))]
  names(p) <- c("Method", "Rate", "Delay", "Cong", "N")[seq_along(p)]
  p
}

# T2: geospatial scale (Small vs Medium) + delta columns
scale_table <- function(df, methods = NULL) {
  sub <- if (!is.null(methods)) df[df$method %in% methods, ] else df
  sub <- sub[!is.na(sub$size), ]
  sub$method <- droplevels(factor(sub$method))
  qmet <- pick_metric(sub, "throughput_rate")
  lmet <- pick_metric(sub, "latency_mean")
  imet <- pick_metric(sub, c("mean_bpr_along_route", "mean_congestion"))
  levs <- intersect(c("Small", "Medium"), unique(as.character(sub$size)))
  ab <- c(Small = "S", Medium = "M")
  meth <- levels(sub$method)
  r <- data.frame(Method = meth)
  for (lv in levs) {
    s2 <- sub[as.character(sub$size) == lv, ]
    s2$method <- droplevels(factor(s2$method))
    p <- ab[[lv]]
    r[[paste0(p, "_Rate")]] <- round(tapply(s2[[qmet]], s2$method, mean, na.rm = TRUE)[meth], 3)
    r[[paste0(p, "_Del")]]  <- round(tapply(s2[[lmet]], s2$method, mean, na.rm = TRUE)[meth], 0)
    r[[paste0(p, "_Cong")]] <- round(tapply(s2[[imet]], s2$method, mean, na.rm = TRUE)[meth], 2)
  }
  if (all(c("Small", "Medium") %in% levs)) {
    r$dRate <- round(100 * (r$M_Rate - r$S_Rate) / r$S_Rate, 1)
    r$dDel  <- round(100 * (r$M_Del  - r$S_Del)  / r$S_Del, 1)
    r$dCong <- round(100 * (r$M_Cong - r$S_Cong) / r$S_Cong, 1)
  }
  r
}

# T3: generalization, Training vs Generalization (proposed methods)
generalization_table <- function(df, methods = PROPOSED_METHODS) {
  sub <- df[df$method %in% methods & !is.na(df$city_group), ]
  sub$method <- droplevels(factor(sub$method))
  qmet <- pick_metric(sub, "throughput_rate")
  lmet <- pick_metric(sub, "latency_mean")
  imet <- pick_metric(sub, c("mean_bpr_along_route", "mean_congestion"))
  if (!length(unique(sub$city_group)) || !nrow(sub)) { message("generalization_table: no data"); return(invisible(NULL)) }
  meth <- levels(sub$method)
  agg <- function(met, grp) {
    s2 <- sub[sub$city_group == grp, ]
    tapply(s2[[met]], droplevels(factor(s2$method)), mean, na.rm = TRUE)[meth]
  }
  tr_thr <- agg(qmet, "Training"); ge_thr <- agg(qmet, "Generalization")
  r <- data.frame(
    Method    = meth,
    Tr_Rate   = round(tr_thr, 3),
    Ge_Rate   = round(ge_thr, 3),
    dRate_pct = round(100 * (ge_thr - tr_thr) / tr_thr, 1),
    Tr_Del    = round(agg(lmet, "Training"), 0),
    Ge_Del    = round(agg(lmet, "Generalization"), 0),
    row.names = NULL)
  r
}

# T4: winning method per scenario facet
winner_by_facets <- function(df, facets = c("task_profile", "cong_profile", "agent_multiplier"),
                             metrics = NULL) {
  if (is.null(metrics)) metrics <- base_metrics(df)
  rows <- list()
  for (fa in facets) {
    if (!fa %in% names(df)) next
    levs <- levels(droplevels(factor(df[[fa]])))
    for (lv in levs) {
      sub <- df[!is.na(df[[fa]]) & as.character(df[[fa]]) == lv, ]
      row <- list(Facet = facet_label(fa), Level = lv)
      for (nm in names(metrics)) {
        met <- metrics[[nm]]
        mu <- tapply(sub[[met]], droplevels(factor(sub$method)), mean, na.rm = TRUE)
        mu <- mu[!is.na(mu)]
        row[[nm]] <- if (!length(mu)) NA_character_ else {
          if (metric_dir(met) == 1) names(mu)[which.max(mu)] else names(mu)[which.min(mu)]
        }
      }
      rows[[length(rows) + 1]] <- row
    }
  }
  do.call(rbind.data.frame, c(rows, stringsAsFactors = FALSE))
}

# ============================================================================
# TEXT SUMMARY (bonus, not counted toward plots/tables)
# ============================================================================
narrative_report <- function(df, by = "scenario", file = NULL,
                             detail_metrics = c("mean_bpr_along_route", "mean_congestion",
                                                "distance_per_task", "mean_extra_steps_per_task"),
                             min_gap_pct = 5) {
  qmet <- pick_metric(df, "throughput_rate")
  scens <- sort(unique(as.character(df[[by]])))
  lines <- c(paste0("=== Automatic summary by ", facet_label(by), " (computed from loaded data) ==="), "")
  for (sc in scens) {
    sub <- df[as.character(df[[by]]) == sc, ]
    thr <- tapply(sub[[qmet]], droplevels(factor(sub$method)), mean, na.rm = TRUE)
    thr <- sort(thr[!is.na(thr)], decreasing = TRUE)
    if (!length(thr)) next
    lines <- c(lines, sprintf("-- %s: %s --", facet_label(by), sc))
    if (length(thr) >= 2) {
      winner <- names(thr)[1]; runner <- names(thr)[2]
      margin <- 100 * (thr[[1]] - thr[[2]]) / abs(thr[[2]])
      lines <- c(lines, sprintf("  Throughput: %s leads (%.3f) ahead of %s (%.3f)  [+%.1f%%].",
                                winner, thr[[1]], runner, thr[[2]], margin))
      for (met in detail_metrics) {
        if (!met %in% names(sub)) next
        mv <- tapply(sub[[met]], droplevels(factor(sub$method)), mean, na.rm = TRUE)
        if (!all(c(winner, runner) %in% names(mv))) next
        wv <- mv[[winner]]; rv <- mv[[runner]]
        if (is.na(wv) || is.na(rv) || rv == 0) next
        d <- 100 * (wv - rv) / abs(rv)
        if (abs(d) < min_gap_pct) next
        favorable <- (metric_dir(met) == 1 && d > 0) || (metric_dir(met) == -1 && d < 0)
        lines <- c(lines, sprintf("    - %s: %s is %s by %.1f%% vs %s -> %s.",
                                  metric_label(met), winner, if (d > 0) "higher" else "lower", abs(d), runner,
                                  if (favorable) "extra advantage" else "trade-off"))
      }
    } else {
      lines <- c(lines, sprintf("  Only one method available: %s (%.3f).", names(thr)[1], thr[[1]]))
    }
    lines <- c(lines, "")
  }
  txt <- paste(lines, collapse = "\n")
  cat(txt, "\n")
  if (!is.null(file)) {
    dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    writeLines(lines, file); message("text -> ", normalizePath(file, winslash = "/"))
  }
  invisible(lines)
}

# ============================================================================
# ORCHESTRATOR
# ============================================================================
run_full_report <- function(root = EVAL_ROOT, dir = PLOT_DIR) {
  df <- load_eval(root)
  df <- add_derived(df)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  f <- function(...) file.path(dir, ...)
  
  ## -- 01_core: UNCHANGED, outside the budget ---------------------------------
  d1 <- f("01_core"); dir.create(d1, recursive = TRUE, showWarnings = FALSE)
  plot_table(perf_table(df), title = "Overall method comparison",
             file = file.path(d1, "comparison_table.png"))
  hist_perf(df, "throughput_rate", file = file.path(d1, "throughput_overall.png"))
  curve_rm(df, "throughput_rate", file = file.path(d1, "throughput_vs_load_rm.png"))
  curve_scale(df, "throughput_rate", file = file.path(d1, "throughput_vs_city_scale.png"))
  progress_eval(df, file = file.path(d1, "evaluation_progress.png"))
  
  ## -- P1: multi-criteria overview (reference RM = 1.0 if available) ---------
  d2 <- f("02_overview"); dir.create(d2, recursive = TRUE, showWarnings = FALSE)
  ref <- if (1 %in% df$rm) df[df$rm == 1, ] else df
  trio_bar(ref, file = file.path(d2, "P1_overview_all_methods.png"),
           title = "Multi-criteria overview (Throughput / Latency / Congestion)  --  RM=1.0")
  
  ## -- P2: load scaling (RM) ----------------------------------------------------
  d3 <- f("03_scale_rm"); dir.create(d3, recursive = TRUE, showWarnings = FALSE)
  rm_trio(df, file = file.path(d3, "P2_load_scaling_RM.png"))
  
  ## -- P3-P5: cross-scenario comparison, one facet at a time ------------------
  d4 <- f("04_scenario_comparison"); dir.create(d4, recursive = TRUE, showWarnings = FALSE)
  qmet <- pick_metric(df, "throughput_rate")
  if ("task_profile" %in% names(df) && nlevels(droplevels(df$task_profile)) > 1)
    facet_bar(df, qmet, "task_profile", file = file.path(d4, "P3_throughput_by_task_profile.png"),
              title = "Throughput by task profile")
  if ("cong_profile" %in% names(df) && nlevels(droplevels(df$cong_profile)) > 1)
    facet_bar(df, qmet, "cong_profile", file = file.path(d4, "P4_throughput_by_congestion_profile.png"),
              title = "Throughput by congestion profile")
  if ("agent_multiplier" %in% names(df) && nlevels(droplevels(df$agent_multiplier)) > 1)
    facet_bar(df, qmet, "agent_multiplier", file = file.path(d4, "P5_throughput_by_agent_multiplier.png"),
              title = "Throughput by agent multiplier (AM)")
  
  ## -- P6: geospatial generalization (proposed methods only) -----------------
  d5 <- f("05_geo_generalization"); dir.create(d5, recursive = TRUE, showWarnings = FALSE)
  if ("city_group" %in% names(df) && length(unique(na.omit(df$city_group))) > 1)
    trio_bar(df, by = "city_group", methods = PROPOSED_METHODS,
             file = file.path(d5, "P6_generalization_training_vs_unseen_cities.png"),
             title = "Geospatial generalization: training vs unseen cities")
  
  ## -- P7: 3 small, readable heatmaps (one per scenario facet) ---------------
  d6 <- f("06_scenario_heatmap"); dir.create(d6, recursive = TRUE, showWarnings = FALSE)
  heatmap_facet_panel(df, qmet, c("task_profile", "cong_profile", "agent_multiplier"),
                      file = file.path(d6, "P7_scenario_comparison_heatmaps.png"))
  
  ## -- P8: geospatial scale sensitivity (Small -> Medium, %-change view) -----
  d7 <- f("07_geo_scale"); dir.create(d7, recursive = TRUE, showWarnings = FALSE)
  delta_trio(df, by = "size", ref = "Small", target = "Medium",
             file = file.path(d7, "P8_scale_sensitivity_small_to_medium.png"),
             title = "Geospatial scale sensitivity: Small -> Medium (% change per method)")
  
  ## -- Tables T1-T5 ---------------------------------------------------------------
  d8 <- f("08_tables"); dir.create(d8, recursive = TRUE, showWarnings = FALSE)
  plot_table(overview_table(df), title = "T1 - Overall Comparison",
             file = file.path(d8, "T1_overall_comparison.png"))
  plot_table(scale_table(df), title = "T2 - Geospatial Scale (Small vs Medium)",
             file = file.path(d8, "T2_geospatial_scale.png"))
  gt <- generalization_table(df)
  if (!is.null(gt)) plot_table(gt, title = "T3 - Generalization (Train vs Unseen)",
                               file = file.path(d8, "T3_generalization.png"))
  plot_table(winner_by_facets(df), title = "T4 - Winning Methods per Scenario",
             file = file.path(d8, "T4_scenario_winners.png"))
  
  ## -- Notes (bonus) --------------------------------------------------------------
  d9 <- f("09_notes"); dir.create(d9, recursive = TRUE, showWarnings = FALSE)
  narrative_report(df, by = "task_profile", file = file.path(d9, "notes_task_profile.txt"))
  narrative_report(df, by = "cong_profile", file = file.path(d9, "notes_congestion_profile.txt"))
  narrative_report(df, by = "agent_multiplier", file = file.path(d9, "notes_agent_multiplier.txt"))
  
  message("\n=== Full report regenerated -> ", normalizePath(dir, winslash = "/"), " ===")
  invisible(df)
}

## -- Auto-run when invoked as `Rscript analysis_extended.R`
if (!interactive() && sys.nframe() == 0) {
  run_full_report()
}