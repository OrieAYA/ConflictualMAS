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
    png(file, width = 90 * p + 260, height = 46 * n + 90, res = 120)
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
    png(file, width = if (is.null(by)) 1100 else 1500, height = 720, res = 120)
  }
  op <- par(mar = c(if (is.null(by)) 6.5 else 7.5, 4.5, 2.5, 1))
  top <- max(mu + if (err) ifelse(is.na(sdv), 0, sdv) else 0, na.rm = TRUE)
  bp <- barplot(mu, beside = TRUE, col = cols, border = NA, las = las,
                ylim = c(0, top * 1.12), ylab = metric,
                main = paste(metric, if (is.null(by)) "" else paste("par", by)),
                cex.names = .8,
                legend.text = if (!is.null(by)) rownames(mu) else NULL,
                args.legend = if (!is.null(by))
                  list(x = "topright", bty = "n", cex = .7, border = NA) else NULL)
  if (err) {
    s <- ifelse(is.na(sdv), 0, sdv)
    suppressWarnings(arrows(bp, mu - s, bp, mu + s, angle = 90, code = 3,
                            length = .03, col = "grey30", lwd = 1))
  }
  if (is.null(by))
    text(bp, mu + s * 0 + top * .04, format(round(mu, 3), nsmall = 3), cex = .72)
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
    png(file, width = 1250, height = 90 + 52 * nC, res = 120)
  }
  op <- par(mar = c(3.2, 9.5, 3.2, 5.5))
  plot.new(); plot.window(c(0, nR), c(nC + .6, .4))
  title(main = sprintf("Avancement evaluation  (RM %s -> %s)",
                       format(min(rm_levels), nsmall = 1),
                       format(max(rm_levels), nsmall = 1)))
  for (i in seq_len(nC)) {
    for (j in seq_len(nR)) {
      rect(j - 1 + .02, i - .34, j - .02, i + .34, col = "grey93", border = NA)
      if (frac[i, j] > 0)
        rect(j - 1 + .02, i - .34, j - 1 + .02 + (j - .04 - (j - 1)) * frac[i, j],
             i + .34, col = "#1b6ca8", border = NA)
      if (frac[i, j] > 0 && frac[i, j] < 1)
        text(j - .5, i, paste0(round(100 * frac[i, j]), "%"),
             cex = .62, col = "grey20")
    }
  }
  axis(2, at = seq_len(nC), labels = cities, las = 1, tick = FALSE, cex.axis = .8)
  axis(3, at = seq_len(nR) - .5, labels = paste0("RM ", colnames(frac)),
       tick = FALSE, cex.axis = .8, line = -.8)
  abline(v = 0:nR, col = "grey75")
  tot <- rowMeans(frac)
  mtext(paste0(round(100 * tot), "%"), side = 4, at = seq_len(nC),
        las = 1, cex = .78, line = .6,
        col = ifelse(tot >= 1, "#2e9e5b", "grey25"))
  mtext(sprintf("global %d%%", round(100 * mean(frac))), side = 1, line = 1.4,
        cex = .85, font = 2)
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
    png(file, width = 1100, height = 700, res = 120)
  }
  cols <- METHOD_COL[rownames(m)]
  if (length(x) == 1) {           # un seul niveau : barres triees (plus lisible)
    o <- order(m[, 1], decreasing = TRUE)
    op <- par(mar = c(4.2, 7.5, 2.5, 1))
    bp <- barplot(rev(m[o, 1]), horiz = TRUE, las = 1, col = rev(cols[o]),
                  border = NA, xlab = metric,
                  main = paste0(metric, "  (RM=", format(x, nsmall = 1), ")"))
    text(rev(m[o, 1]), bp, labels = format(round(rev(m[o, 1]), 3), nsmall = 3),
         pos = 2, cex = .75, col = "white", font = 2)
    par(op)
    if (to_file) { dev.off(); message("plot -> ", normalizePath(file, winslash = "/")) }
    message("un seul niveau RM : barres affichees ; la courbe apparaitra des RM>=2 niveaux")
    return(as.data.frame(round(m, 4)))
  }
  op <- par(mar = c(4.2, 4.2, 2.5, 1))
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
         bty = "n", cex = .75, ncol = 4)
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
    png(file, width = 400 + 320 * length(panels), height = 700, res = 120)
  }
  op <- par(mfrow = c(1, length(panels)), mar = c(4.2, 4.2, 2.5, 1))
  res <- list()
  for (p in panels) {
    sub <- if (is.na(p)) df else df[df$rm == p, ]
    m <- mat_by(sub, metric, "size")
    m <- m[, colSums(!is.na(m)) > 0, drop = FALSE]
    res[[if (is.na(p)) "all" else paste0("rm", p)]] <- as.data.frame(round(m, 4))
    cols <- METHOD_COL[rownames(m)]
    yr <- range(m, na.rm = TRUE)
    matplot(seq_len(ncol(m)), t(m), type = "b", pch = 16, lty = 1, lwd = 2,
            col = cols, xaxt = "n", xlim = c(0.8, ncol(m) + 0.2),
            ylim = yr + c(-.28, .06) * max(diff(yr), 1e-9),
            xlab = "Echelle d'environnement", ylab = metric,
            main = if (is.na(p)) paste(metric, "vs echelle")
                   else paste0("RM=", format(p, nsmall = 1)))
    axis(1, at = seq_len(ncol(m)), labels = colnames(m))
    grid(col = "grey85")
    legend("bottom", rownames(m), col = cols, lty = 1, lwd = 2, pch = 16,
           bty = "n", cex = .7, ncol = 4)
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
