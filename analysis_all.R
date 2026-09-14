# ============================================================================
# analysis_all.R -- fichier unique (eval + extended + paper report)
# ----------------------------------------------------------------------------
# Fusion de analysis_eval.R + analysis_extended.R + analysis_paper.R.
# Usage : ouvrir ce fichier, Ctrl+A puis Run (ou Rscript analysis_all.R).
#
# Ce que ca lance automatiquement a la fin du fichier :
#   - run_paper_report()  -> TOUJOURS execute (main + supplementary, Figure 1/2,
#                            Table 1/2, Table S1-S3, Figure S2-S4)
#   - run_full_report()   -> seulement si lance via 'Rscript analysis_all.R'
#                            (pas en interactif / Ctrl+A) ; sinon appelle-la
#                            toi-meme : run_full_report()
# ============================================================================

# Analyse des evaluations : results/paper_eval/pol{P}_rm{RM}_g{G}/
#   episodes_seed*.csv              -> MAPPO IPPO MAPPER Hybrid RMCA TokenPassing
#   sota_standalone/sota_seed*.csv  -> CA (FaithfulCongestionAware), HAPC
#
# Usage :
#   res <- report()            # table + 2 graphiques PNG dans results/paper_eval/plots/
#   res$perf ; res$rm ; res$scale
#   curve_rm(df, "latency_mean")    # meme courbe sur une autre metrique

EVAL_SUBDIR <- "results/paper_eval"

# Racine projet resolue automatiquement : repertoire courant puis parents.
find_project_root <- function(start = getwd()) {
  d <- normalizePath(start, winslash = "/", mustWork = FALSE)
  for (i in 1:8) {
    if (dir.exists(file.path(d, EVAL_SUBDIR))) return(d)
    p <- dirname(d)
    if (identical(p, d)) break
    d <- p
  }
  if (dir.exists(file.path("C:/ConflictualMAS", EVAL_SUBDIR))) return("C:/ConflictualMAS")
  NA_character_
}

# Forcer la racine a la main si l'auto-detection echoue : set_root("C:/ConflictualMAS")
set_root <- function(root) {
  root <<- normalizePath(root, winslash = "/", mustWork = TRUE)
  PROJECT_ROOT <<- root
  EVAL_ROOT    <<- file.path(root, EVAL_SUBDIR)
  PLOT_DIR     <<- file.path(EVAL_ROOT, "plots")
  message("racine projet : ", PROJECT_ROOT)
  invisible(root)
}

PROJECT_ROOT <- find_project_root()
if (is.na(PROJECT_ROOT)) {
  EVAL_ROOT <- EVAL_SUBDIR
  PLOT_DIR  <- file.path(EVAL_ROOT, "plots")
  warning("racine projet introuvable depuis ", getwd(),
          " -- appelle set_root(\"C:/ConflictualMAS\")", call. = FALSE)
} else {
  EVAL_ROOT <- file.path(PROJECT_ROOT, EVAL_SUBDIR)
  PLOT_DIR  <- file.path(EVAL_ROOT, "plots")
  message("racine projet : ", PROJECT_ROOT)
}

METHOD_ORDER <- c("MAPPO", "IPPO", "MAPPER", "Hybrid", "RMCA", "TokenPassing", "CA", "HAPC")
METHOD_COL   <- c(MAPPO = "#1b6ca8", IPPO = "#2e9e5b", MAPPER = "#8e5bb5",
                  Hybrid = "#d1495b", RMCA = "#e08a1e", TokenPassing = "#6b7280",
                  CA = "#00a6a6", HAPC = "#a15c00")
# distinct line type / point shape per method -- helps tell curves apart when
# two methods have near-identical values and their lines visually overlap
METHOD_LTY <- setNames(rep(1:6, length.out = length(METHOD_ORDER)), METHOD_ORDER)
METHOD_PCH <- setNames(rep(c(16, 17, 15, 18, 8, 4), length.out = length(METHOD_ORDER)), METHOD_ORDER)

KEEP <- c("tasks_appeared", "tasks_completed", "throughput_rate", "latency_mean",
          "latency_per_agent", "mean_wait_steps", "mean_trip_steps",
          "agent_utilisation", "agent_completed_gini", "mean_congestion",
          "mean_bpr_along_route", "route_congestion_exposure",
          "n_traversals_in_jam", "delivery_route_efficiency",
          "mean_extra_steps_per_task", "total_fleet_distance_m",
          "compute_time_per_task_ms", "wallclock_ms",
          "capacity_violations", "pairing_violations")

load_eval <- function(root = EVAL_ROOT) {
  dirs <- list.dirs(root, recursive = FALSE)
  out <- list()
  for (d in dirs) {
    # tag = any target label: g1 (old groups), Paris_Medium, Paris, all, ...
    m <- regmatches(basename(d),
                    regexec("^pol(\\d+)_rm([0-9.]+)_(.+)$", basename(d)))[[1]]
    if (length(m) != 4) next
    pol <- as.integer(m[2]); rm_lvl <- as.numeric(m[3]); grp <- m[4]
    
    for (f in list.files(d, "^episodes_seed.*\\.csv$", full.names = TRUE)) {
      x <- read.csv(f, stringsAsFactors = FALSE)
      if (!nrow(x)) next
      x <- x[!is.na(x$throughput_rate) & nzchar(x$policy_mode), ]
      if (!nrow(x)) next
      d1 <- data.frame(method   = x$policy_mode,
                       city     = x$city,
                       scenario = sub("^eval_", "", x$phase),
                       n_agents = x$n_agents_max,
                       stringsAsFactors = FALSE)
      for (k in KEEP) d1[[k]] <- if (k %in% names(x)) x[[k]] else NA_real_
      d1$rm <- rm_lvl; d1$group <- grp; d1$policy_seed <- pol; d1$family <- "pipeline"
      out[[length(out) + 1]] <- d1
    }
    
    for (f in list.files(file.path(d, "sota_standalone"),
                         "^sota_seed.*\\.csv$", full.names = TRUE)) {
      x <- read.csv(f, stringsAsFactors = FALSE)
      if (!nrow(x)) next
      x <- x[!is.na(x$throughput_rate) & nzchar(x$solver), ]
      if (!nrow(x)) next
      lbl <- c(FaithfulCongestionAware = "CA", HybridAdaptivePredictive = "HAPC")
      d2 <- data.frame(method   = ifelse(x$solver %in% names(lbl), lbl[x$solver], x$solver),
                       city     = x$city,
                       scenario = x$scenario,
                       n_agents = x$n_agents,
                       stringsAsFactors = FALSE)
      for (k in KEEP) d2[[k]] <- if (k %in% names(x)) x[[k]] else NA_real_
      d2$rm <- rm_lvl; d2$group <- grp; d2$policy_seed <- pol; d2$family <- "standalone"
      out[[length(out) + 1]] <- d2
    }
  }
  if (!length(out)) stop("aucune donnee sous ", root)
  df <- do.call(rbind, out)
  df$size <- factor(ifelse(grepl("_Large$", df$city), "Large",
                           ifelse(grepl("_Medium$", df$city), "Medium", "Small")),
                    levels = c("Small", "Medium", "Large"))
  df$town   <- sub("_(Small|Medium|Large)$", "", df$city)
  df$method <- factor(df$method, levels = intersect(METHOD_ORDER, unique(df$method)))
  df
}

# Matrice methode x <by> d'une metrique (moyenne)
mat_by <- function(df, metric, by, fun = mean) {
  m <- tapply(df[[metric]], list(df$method, df[[by]]), fun, na.rm = TRUE)
  m[rowSums(!is.na(m)) > 0, , drop = FALSE]
}

# ── RETURN 1 : performances globales ────────────────────────────────────────
perf_table <- function(df) {
  s <- function(metric, f) tapply(df[[metric]], df$method, f, na.rm = TRUE)
  r <- data.frame(
    Method      = levels(droplevels(df$method)),
    Thr         = round(s("throughput_rate", mean), 3),
    Thr_sd      = round(s("throughput_rate", sd), 3),
    Lat         = round(s("latency_mean", mean), 0),
    BPR         = round(s("mean_bpr_along_route", mean), 2),
    Cong        = round(s("mean_congestion", mean), 2),
    Gini        = round(s("agent_completed_gini", mean), 3),
    ms_per_task = round(s("compute_time_per_task_ms", mean), 0),
    N           = as.integer(table(droplevels(df$method))),
    row.names = NULL, stringsAsFactors = FALSE)
  r[order(-r$Thr), ]
}

# ── Rendu graphique d'une table (image exploitable pour l'article) ─────────
plot_table <- function(tab, title = NULL, cex = 1, bold_rows = 1, file = NULL) {
  tab <- as.data.frame(tab, stringsAsFactors = FALSE)
  n <- nrow(tab); p <- ncol(tab)
  cells <- vapply(seq_len(p), function(j) format(tab[[j]], trim = TRUE),
                  character(n))
  if (is.null(dim(cells))) cells <- matrix(cells, nrow = n)
  hdr <- names(tab)
  w <- vapply(seq_len(p), function(j) max(nchar(c(hdr[j], cells[, j]))) + 2, 0)
  xr <- cumsum(c(0, w / sum(w)))
  xc <- xr[-1] - 0.004
  
  to_file <- !is.null(file)
  if (to_file) {
    dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    png(file, width = 108 * p + 320, height = 56 * n + 110, res = 120, pointsize = 15)
  }
  op <- par(mar = c(0.4, 0.4, if (is.null(title)) 0.4 else 2.2, 0.4))
  plot.new(); plot.window(c(0, 1), c(n + 0.7, -1.1))
  if (!is.null(title)) title(main = title, cex.main = cex * 1.05)
  for (i in seq_len(n)) if (i %% 2 == 0)
    rect(0, i - .5, 1, i + .5, col = "grey96", border = NA)
  text(xc, 0, hdr, adj = c(1, .5), font = 2, cex = cex)
  segments(0, -.5, 1, -.5); segments(0, .5, 1, .5)
  for (i in seq_len(n))
    text(xc, i, cells[i, ], adj = c(1, .5), cex = cex,
         font = if (i %in% bold_rows) 2 else 1)
  segments(0, n + .5, 1, n + .5)
  par(op)
  if (to_file) { dev.off(); message("table -> ", normalizePath(file, winslash = "/")) }
  invisible(tab)
}

# ── Histogramme (barres verticales + ecart-type) ────────────────────────────
# by = NULL -> une barre par methode ; by = "scenario"/"size"/"rm" -> groupes
hist_perf <- function(df, metric = "throughput_rate", by = NULL,
                      err = TRUE, file = NULL, las = 1) {
  if (is.null(by)) {
    mu <- tapply(df[[metric]], df$method, mean, na.rm = TRUE)
    sdv <- tapply(df[[metric]], df$method, sd, na.rm = TRUE)
    o <- order(mu, decreasing = TRUE)
    mu <- mu[o]; sdv <- sdv[o]
    cols <- METHOD_COL[names(mu)]
  } else {
    mu <- mat_by(df, metric, by)
    sdv <- mat_by(df, metric, by, sd)
    cols <- METHOD_COL[rownames(mu)]
  }
  to_file <- !is.null(file)
  if (to_file) {
    dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    png(file, width = if (is.null(by)) 1150 else 1600, height = 760, res = 120, pointsize = 15)
  }
  op <- par(mar = c(if (is.null(by)) 7.5 else 8.5, 5.2, 3, 1), cex.axis = 1, cex.lab = 1.15, cex.main = 1.2)
  top <- max(mu + if (err) ifelse(is.na(sdv), 0, sdv) else 0, na.rm = TRUE)
  bp <- barplot(mu, beside = TRUE, col = cols, border = NA, las = las,
                ylim = c(0, top * 1.12), ylab = metric,
                main = paste(metric, if (is.null(by)) "" else paste("par", by)),
                cex.names = .95,
                legend.text = if (!is.null(by)) rownames(mu) else NULL,
                args.legend = if (!is.null(by))
                  list(x = "topright", bty = "n", cex = .85, border = NA) else NULL)
  if (err) {
    s <- ifelse(is.na(sdv), 0, sdv)
    suppressWarnings(arrows(bp, mu - s, bp, mu + s, angle = 90, code = 3,
                            length = .03, col = "grey30", lwd = 1))
  }
  if (is.null(by))
    text(bp, mu + s * 0 + top * .04, format(round(mu, 3), nsmall = 3), cex = .85)
  abline(h = 0, col = "grey40")
  par(op)
  if (to_file) { dev.off(); message("plot -> ", normalizePath(file, winslash = "/")) }
  invisible(round(mu, 4))
}

# ── Avancement du sweep RM, une barre de progression par ville ──────────────
PROTO_CITIES <- as.vector(outer(c("Tokyo", "Kyoto", "LosAngeles", "NewYork", "Paris"),
                                c("Small", "Medium"), paste, sep = "_"))

# show_pending = TRUE (defaut) : inclut les villes du protocole pas encore
# demarrees, sinon le pourcentage global est biaise vers le haut.
progress_eval <- function(df, rm_levels = seq(1, 2.5, by = .5),
                          n_scen = 27, n_meth = NULL, cities = NULL,
                          show_pending = TRUE, file = NULL) {
  if (is.null(n_meth)) n_meth <- max(2, nlevels(droplevels(df$method)))
  expected <- n_scen * n_meth
  if (is.null(cities))
    cities <- if (show_pending) union(PROTO_CITIES, unique(df$city))
  else unique(df$city)
  cities <- sort(cities)
  nC <- length(cities); nR <- length(rm_levels)
  done <- matrix(0, nC, nR, dimnames = list(cities, format(rm_levels, nsmall = 1)))
  for (i in seq_len(nC)) for (j in seq_len(nR))
    done[i, j] <- sum(df$city == cities[i] & df$rm == rm_levels[j])
  frac <- pmin(done / expected, 1)
  
  to_file <- !is.null(file)
  if (to_file) {
    dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    png(file, width = 1350, height = 110 + 56 * nC, res = 120, pointsize = 15)
  }
  op <- par(mar = c(3.6, 10.5, 3.6, 6))
  plot.new(); plot.window(c(0, nR), c(nC + .6, .4))
  title(main = sprintf("Avancement evaluation  (RM %s -> %s)",
                       format(min(rm_levels), nsmall = 1),
                       format(max(rm_levels), nsmall = 1)), cex.main = 1)
  for (i in seq_len(nC)) {
    for (j in seq_len(nR)) {
      rect(j - 1 + .02, i - .34, j - .02, i + .34, col = "grey93", border = NA)
      if (frac[i, j] > 0)
        rect(j - 1 + .02, i - .34, j - 1 + .02 + (j - .04 - (j - 1)) * frac[i, j],
             i + .34, col = "#1b6ca8", border = NA)
      if (frac[i, j] > 0 && frac[i, j] < 1)
        text(j - .5, i, paste0(round(100 * frac[i, j]), "%"),
             cex = .74, col = "grey20")
    }
  }
  axis(2, at = seq_len(nC), labels = cities, las = 1, tick = FALSE, cex.axis = .95)
  axis(3, at = seq_len(nR) - .5, labels = paste0("RM ", colnames(frac)),
       tick = FALSE, cex.axis = .95, line = -.8)
  abline(v = 0:nR, col = "grey75")
  tot <- rowMeans(frac)
  mtext(paste0(round(100 * tot), "%"), side = 4, at = seq_len(nC),
        las = 1, cex = .92, line = .6,
        col = ifelse(tot >= 1, "#2e9e5b", "grey25"))
  mtext(sprintf("global %d%%", round(100 * mean(frac))), side = 1, line = 1.6,
        cex = 1, font = 2)
  par(op)
  if (to_file) { dev.off(); message("plot -> ", normalizePath(file, winslash = "/")) }
  invisible(round(frac, 3))
}

# ── RETURN 2 : courbe de reponse a la charge (incrementations de RM) ────────
curve_rm <- function(df, metric = "throughput_rate", file = NULL, ylab = NULL) {
  m <- mat_by(df, metric, "rm")
  x <- as.numeric(colnames(m))
  to_file <- !is.null(file)
  if (to_file) {
    dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    png(file, width = 1200, height = 770, res = 120, pointsize = 16)
  }
  cols <- METHOD_COL[rownames(m)]
  if (length(x) == 1) {           # un seul niveau : barres triees (plus lisible)
    o <- order(m[, 1], decreasing = TRUE)
    op <- par(mar = c(4.8, 9, 3.8, 1), cex.axis = 1.05, cex.lab = 1.3, cex.main = 1.55)
    bp <- barplot(rev(m[o, 1]), horiz = TRUE, las = 1, col = rev(cols[o]),
                  border = NA, xlab = metric,
                  main = paste0(metric, "  (RM=", format(x, nsmall = 1), ")"))
    text(rev(m[o, 1]), bp, labels = format(round(rev(m[o, 1]), 3), nsmall = 3),
         pos = 2, cex = .95, col = "white", font = 2)
    par(op)
    if (to_file) { dev.off(); message("plot -> ", normalizePath(file, winslash = "/")) }
    message("un seul niveau RM : barres affichees ; la courbe apparaitra des RM>=2 niveaux")
    return(as.data.frame(round(m, 4)))
  }
  op <- par(mar = c(4.8, 5.4, 3.8, 1), cex.axis = 1.05, cex.lab = 1.3, cex.main = 1.55)
  yr <- range(m, na.rm = TRUE)
  matplot(x, t(m), type = "b", pch = 16, lty = 1, lwd = 2, col = cols,
          xlab = "Ratio multiplier RM (charge d'evenements)",
          ylab = if (is.null(ylab)) metric else ylab,
          main = paste(metric, "vs charge"), xaxt = "n",
          xlim = if (length(x) > 1) range(x) else x + c(-.5, .5),
          ylim = yr + c(-.28, .06) * max(diff(yr), 1e-9))
  axis(1, at = x, labels = format(x, nsmall = 1))
  grid(col = "grey85")
  legend("bottom", rownames(m), col = cols, lty = 1, lwd = 2, pch = 16,
         bty = "n", cex = 1, ncol = 4)
  par(op)
  if (to_file) { dev.off(); message("plot -> ", normalizePath(file, winslash = "/")) }
  if (length(x) < 2)
    message("un seul niveau RM pour l'instant : la courbe s'etoffera aux RM suivants")
  as.data.frame(round(m, 4))
}

# ── RETURN 3 : performances en fonction de l'echelle d'environnement ────────
curve_scale <- function(df, metric = "throughput_rate", file = NULL, by_rm = TRUE) {
  rms <- sort(unique(df$rm))
  panels <- if (by_rm && length(rms) > 1) rms else NA
  to_file <- !is.null(file)
  if (to_file) {
    dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    png(file, width = 460 + 360 * length(panels), height = 770, res = 120, pointsize = 16)
  }
  op <- par(mfrow = c(1, length(panels)), mar = c(4.8, 5.4, 3.8, 1),
            cex.axis = 1.05, cex.lab = 1.3, cex.main = 1.55)
  res <- list()
  for (p in panels) {
    sub <- if (is.na(p)) df else df[df$rm == p, ]
    m <- mat_by(sub, metric, "size")
    m <- m[, colSums(!is.na(m)) > 0, drop = FALSE]
    res[[if (is.na(p)) "all" else paste0("rm", p)]] <- as.data.frame(round(m, 4))
    cols <- METHOD_COL[rownames(m)]
    yr <- range(m, na.rm = TRUE)
    leg_n <- nrow(m); leg_ncol <- if (leg_n <= 3) leg_n else 2
    leg_rows <- ceiling(leg_n / leg_ncol)
    matplot(seq_len(ncol(m)), t(m), type = "b", pch = 16, lty = 1, lwd = 2,
            col = cols, xaxt = "n", xlim = c(0.8, ncol(m) + 0.2),
            ylim = yr + c(-.14 * leg_rows - .1, .06) * max(diff(yr), 1e-9),
            xlab = "Echelle d'environnement", ylab = metric,
            main = if (is.na(p)) paste(metric, "vs echelle")
            else paste0("RM=", format(p, nsmall = 1)))
    axis(1, at = seq_len(ncol(m)), labels = colnames(m))
    grid(col = "grey85")
    legend("bottom", rownames(m), col = cols, lty = 1, lwd = 2, pch = 16,
           bty = "n", cex = .8, ncol = leg_ncol)
  }
  par(op)
  if (to_file) { dev.off(); message("plot -> ", normalizePath(file, winslash = "/")) }
  if (length(res) == 1) res[[1]] else res
}

check_integrity <- function(df) {
  cat("== integrite ==\n")
  cat("  violations capacite :", sum(df$capacity_violations, na.rm = TRUE),
      "| pairing :", sum(df$pairing_violations, na.rm = TRUE), "\n")
  k <- paste(df$city, df$scenario, df$rm, df$group)
  bad <- tapply(df$tasks_appeared, k, function(v) length(unique(v)) > 1)
  cat("  slots :", length(bad), "| flux de taches divergent :", sum(bad, na.rm = TRUE), "\n")
  cnt <- table(droplevels(df$method))
  cat("  episodes/methode :", paste(names(cnt), cnt, sep = "=", collapse = "  "), "\n")
}

# Exporte tous les graphiques en PNG dans dir/ (chemins absolus affiches)
save_all <- function(df = NULL, dir = PLOT_DIR, metric = "throughput_rate") {
  if (is.null(df)) df <- load_eval()
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  f <- function(n) file.path(dir, n)
  plot_table(perf_table(df), title = "Performances globales", file = f("perf_table.png"))
  hist_perf(df, metric,            file = f("hist_throughput.png"))
  hist_perf(df, metric, by = "size", file = f("hist_par_echelle.png"))
  curve_rm(df, metric,             file = f("rm_throughput.png"))
  curve_scale(df, metric,          file = f("scale_throughput.png"))
  progress_eval(df,                file = f("progress.png"))
  message("=> ", normalizePath(dir, winslash = "/"))
  invisible(dir)
}

report <- function(root = EVAL_ROOT, save = FALSE) {
  df <- load_eval(root)
  cat("== couverture ==\n")
  cat("  lignes :", nrow(df),
      "| RM :", paste(format(sort(unique(df$rm)), nsmall = 1), collapse = ", "),
      "| groupes :", paste(sort(unique(df$group)), collapse = ", "), "\n")
  cat("  villes :", paste(sort(unique(df$city)), collapse = ", "), "\n\n")
  check_integrity(df)
  
  perf <- perf_table(df)
  cat("\n== performances globales ==\n"); print(perf, row.names = FALSE)
  plot_table(perf, title = "Performances globales",
             file = if (save) file.path(PLOT_DIR, "perf_table.png"))
  
  rm_tab <- curve_rm(df, file = if (save) file.path(PLOT_DIR, "rm_throughput.png"))
  cat("\n== throughput par niveau de charge RM ==\n"); print(rm_tab)
  
  sc_tab <- curve_scale(df, file = if (save) file.path(PLOT_DIR, "scale_throughput.png"))
  cat("\n== throughput par echelle d'environnement ==\n"); print(sc_tab)
  
  invisible(list(df = df, perf = perf, rm = rm_tab, scale = sc_tab))
}
# ============================================================================
# --- Contenu de analysis_extended.R --------------------------------------
# ============================================================================

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
#   (interactif : run_full_report() -- deja definie plus haut dans ce fichier)
# ============================================================================

## -- Load the base script (analysis_eval.R) ---------------------------------
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0 || !nzchar(a)) b else a

# (analysis_eval.R deja charge plus haut dans ce fichier -- pas besoin de le
#  re-sourcer ici)

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
  mtext(title, outer = TRUE, font = 2, cex = 1)
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
  meth    <- rownames(mat_by(df, metrics[[1]], "rm"))
  leg_n   <- length(meth); leg_ncol <- if (leg_n <= 3) leg_n else 3
  leg_rows <- ceiling(leg_n / leg_ncol)

  to_file <- !is.null(file)
  if (to_file) { dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    png(file, width = 840 * length(metrics), height = round(940 + 210 * leg_rows),
        res = 120, pointsize = 22) }

  layout(matrix(c(seq_len(length(metrics)), rep(length(metrics) + 1, length(metrics))),
                nrow = 2, byrow = TRUE),
         heights = c(3, 1))
  op <- par(oma = c(0, 0, 4, 0), mar = c(5.2, 6.6, 4, 1),
            cex.axis = 1.25, cex.lab = 1.5, cex.main = 1.8)
  for (nm in names(metrics)) {
    met <- metrics[[nm]]
    m <- mat_by(df, met, "rm")
    m <- m[, colSums(!is.na(m)) > 0, drop = FALSE]
    if (!ncol(m)) { plot.new(); title(main = paste(nm, "(n/a)")); next }
    x <- as.numeric(colnames(m))
    cols <- METHOD_COL[rownames(m)]
    if (length(x) < 2) {
      o <- order(m[, 1], decreasing = (metric_dir(met) == 1))
      barplot(m[o, 1], col = cols[o], las = 2, cex.names = 1.15, main = nm, ylab = metric_label(met))
    } else {
      yr <- range(m, na.rm = TRUE)
      matplot(x, t(m), type = "b", pch = METHOD_PCH[rownames(m)], cex = 1.35,
              lty = METHOD_LTY[rownames(m)], lwd = 4.2, col = cols, xaxt = "n",
              xlab = "Ratio Multiplier (RM)", ylab = metric_label(met), main = nm,
              ylim = yr + c(-.1, .1) * max(diff(yr), 1e-9))
      axis(1, at = x, labels = format(x, nsmall = 1))
      grid(col = "grey85")
    }
  }
  par(mar = c(0.2, 0.2, 0.2, 0.2))
  plot.new()
  legend("center", meth, col = METHOD_COL[meth], lty = METHOD_LTY[meth], lwd = 4.2,
         pch = METHOD_PCH[meth], pt.cex = 1.35, bty = "n", cex = 1.55, ncol = leg_ncol,
         x.intersp = 1.5, y.intersp = 1.8)
  mtext(title, outer = TRUE, font = 2, cex = 1)
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
        main = main, cex.main = 2, asp = 1)
  axis(1, at = seq_len(nC), labels = colnames(z), las = 1, cex.axis = 1.8)
  if (show_labels) axis(2, at = seq_len(nR), labels = rownames(z), las = 1, cex.axis = 1.25)
  for (i in seq_len(nR)) for (j in seq_len(nC))
    if (!is.na(m[i, j]))
      text(j, i, format(round(m[i, j], 2), nsmall = 2), cex = 1.8, font = 2,
           col = ifelse(z[i, j] > .5, "grey15", "white"))
  box()
  invisible(z)
}

# -- shared method-name column (drawn once, on the far left; centred, no header) --
heatmap_row_labels <- function(meth) {
  nR <- length(meth)
  plot.new(); plot.window(c(0, 1), c(.5, nR + .5))
  text(0.5, seq_len(nR), meth, adj = c(0.5, 0.5), font = 2, cex = 2, xpd = NA)
}

heatmap_facet_panel <- function(df, metric, facets, file = NULL,
                                title = "Cross-scenario comparison  (green = best method, red = worst)") {
  m_ref <- mat_by(df, metric, facets[1])
  meth  <- rownames(m_ref)
  nR    <- length(meth)
  nC    <- max(vapply(facets, function(fa) ncol(mat_by(df, metric, fa)), integer(1)))

  cell_px <- 210                                   # one heatmap cell, in px -- kept square via asp=1
  mar_b <- 5.4; mar_l <- 1; mar_t <- 3.8; mar_r <- 0.3
  line_px <- 18 / 72 * 120                          # ~px per margin "line" at pointsize 18, res 120
  panel_w <- round(nC * cell_px + (mar_l + mar_r) * line_px)
  panel_h <- round(nR * cell_px + (mar_t + mar_b) * line_px)
  label_w <- 340
  oma_top_lines <- 5.6
  oma_top <- round(oma_top_lines * line_px)

  to_file <- !is.null(file)
  if (to_file) { dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    png(file, width = label_w + panel_w * length(facets), height = panel_h + oma_top, res = 120, pointsize = 22) }
  op <- par(oma = c(0, 0, oma_top_lines, 0), mar = c(mar_b, mar_l, mar_t, mar_r))
  layout(matrix(seq_len(length(facets) + 1), nrow = 1),
         widths = c(label_w, rep(panel_w, length(facets))))
  heatmap_row_labels(meth)
  for (fa in facets) heatmap_draw(df, metric, fa, main = facet_label(fa), show_labels = FALSE)
  mtext(title, outer = TRUE, font = 2, cex = 1, line = 2.6)
  mtext(paste0("(", metric_label(metric), ")"), outer = TRUE, font = 3, cex = 1.3, line = 0.9)
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
  mtext(title, outer = TRUE, font = 2, cex = 1)
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
# ============================================================================
# --- Contenu de analysis_paper.R ------------------------------------------
# ============================================================================

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
#   (eval + extended deja charges plus haut dans ce meme fichier)
#   run_paper_report()   # writes everything under EVAL_ROOT/plots/paper/
# ============================================================================

# (analysis_eval.R et analysis_extended.R deja charges plus haut dans ce
#  fichier -- pas besoin de les re-sourcer ici)

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