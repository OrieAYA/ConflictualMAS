library(ggplot2)
library(dplyr)
library(tidyr)

df <- read.csv("C:/ConflictualMAS/scripts/episodes_train.csv")

# ── Preparation : seeds 42, 43, 44 ───────────────────────────────────────
df <- df[df$seed %in% c(42, 43, 44), ]
df$ep_in_seed <- df$global_episode - 162 * (df$seed - 42) + 1

# ── Nombre d'episodes attendu par seed (donnees finales : 162/seed) ─────
# NB : ne PAS deduire ce nombre du CSV via min() -> si une seed est encore
# en cours (training pas termine), ca tronquerait a tort toutes les autres.
n_common <- 162

# Verification / avertissement si une seed n'a pas encore ses 162 episodes
completude <- df %>%
  group_by(seed) %>%
  summarise(n = n_distinct(ep_in_seed), .groups = "drop")
manquantes <- completude %>% filter(n < n_common)
if (nrow(manquantes) > 0) {
  warning(
    "Seed(s) incomplete(s) dans ce CSV : ",
    paste(sprintf("seed %d (%d/%d episodes)", manquantes$seed, manquantes$n, n_common), collapse = ", "),
    ". Ces seeds seront ignorees pour l'instant -> relancer ce script une fois",
    " les donnees finales disponibles pour les inclure."
  )
}

# On ne garde que les seeds qui ont bien leurs 162 episodes complets
seeds_completes <- completude %>% filter(n >= n_common) %>% pull(seed)
df <- df %>% filter(seed %in% seeds_completes, ep_in_seed <= n_common)

# Verification : meme nombre d'episodes par (seed, methode)
# print(df %>% count(seed, policy_mode))

# ── Agregation : moyenne + sd sur les 3 seeds, par methode et episode ───
agg <- df %>%
  group_by(policy_mode, ep_in_seed) %>%
  summarise(
    entropy_m      = mean(entropy),      entropy_sd      = sd(entropy),
    actor_loss_m   = mean(actor_loss),   actor_loss_sd   = sd(actor_loss),
    critic_loss_m  = mean(critic_loss),  critic_loss_sd  = sd(critic_loss),
    adv_std_m      = mean(adv_std),      adv_std_sd      = sd(adv_std),
    .groups = "drop"
  )

theme_set(theme_minimal())

# ══════════════════════════════════════════════════════════════════════
# PLOTS 1-3 : comparaison des 3 methodes (moyenne +/- sd sur les seeds)
# ══════════════════════════════════════════════════════════════════════

# ── Plot 1 : Entropy (exploration) ───────────────────────────────────
p1 <- ggplot(agg, aes(ep_in_seed, entropy_m, color = policy_mode, fill = policy_mode)) +
  geom_ribbon(aes(ymin = entropy_m - entropy_sd, ymax = entropy_m + entropy_sd),
              alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  labs(x = "Episode", y = "Entropy",
       color = "Method", fill = "Method",
       title = "Entropy over training") +
  theme(legend.position = "bottom")
print(p1)

# ── Plot 2 : Critic loss (stabilite de la fonction de valeur) ───────
p2 <- ggplot(agg, aes(ep_in_seed, critic_loss_m, color = policy_mode, fill = policy_mode)) +
  geom_ribbon(aes(ymin = critic_loss_m - critic_loss_sd, ymax = critic_loss_m + critic_loss_sd),
              alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  labs(x = "Episode", y = "Critic loss",
       color = "Method", fill = "Method",
       title = "Critic loss over training") +
  theme(legend.position = "bottom")
print(p2)

# ── Plot 3 : Actor loss (stabilite de la politique) ──────────────────
p3 <- ggplot(agg, aes(ep_in_seed, actor_loss_m, color = policy_mode, fill = policy_mode)) +
  geom_ribbon(aes(ymin = actor_loss_m - actor_loss_sd, ymax = actor_loss_m + actor_loss_sd),
              alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  labs(x = "Episode", y = "Actor loss",
       color = "Method", fill = "Method",
       title = "Actor loss over training") +
  theme(legend.position = "bottom")
print(p3)

# ══════════════════════════════════════════════════════════════════════
# PLOTS 4-6 : vue unique par methode - metriques couplees et normalisees
# (entropy, actor_loss, critic_loss, adv_std ramenes sur [0,1] pour
#  pouvoir les superposer malgre des echelles tres differentes)
# ══════════════════════════════════════════════════════════════════════

norm01 <- function(x) {
  r <- range(x, na.rm = TRUE)
  if (diff(r) == 0) return(rep(0.5, length(x)))
  (x - r[1]) / diff(r)
}

# Pour les losses qui peuvent etre negatives (actor_loss, critic_loss),
# on normalise sur |x| : ainsi "proche de 0" reste toujours en bas (0) et
# "loin de 0" en haut (1), quel que soit le signe -> convergence = descend
# vers 0 pour TOUTES les courbes, comme entropy/adv_std.
norm01_abs <- function(x) norm01(abs(x))

make_method_plot <- function(method_name) {
  sub <- agg %>%
    filter(policy_mode == method_name) %>%
    mutate(
      Entropy       = norm01(entropy_m),
      `Actor loss`  = norm01_abs(actor_loss_m),
      `Critic loss` = norm01_abs(critic_loss_m),
      `Adv std`     = norm01(adv_std_m)
    ) %>%
    select(ep_in_seed, Entropy, `Actor loss`, `Critic loss`, `Adv std`) %>%
    pivot_longer(-ep_in_seed, names_to = "metric", values_to = "value")
  
  ggplot(sub, aes(ep_in_seed, value, color = metric)) +
    geom_line(linewidth = 1) +
    labs(x = "Episode",
         y = "Normalized value [0-1]", color = "Metric",
         title = paste0(method_name, " training metrics")) +
    theme(legend.position = "bottom")
}

# ── Plot 4 : IPPO ─────────────────────────────────────────────────────
p4 <- make_method_plot("IPPO")
print(p4)

# ── Plot 5 : MAPPER ───────────────────────────────────────────────────
p5 <- make_method_plot("MAPPER")
print(p5)

# ── Plot 6 : MAPPO ─────────────────────────────────────────────────────
p6 <- make_method_plot("MAPPO")
print(p6)