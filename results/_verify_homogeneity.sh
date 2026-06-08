#!/bin/bash
# Run AFTER main.exe OP completes.
# Cross-checks tasks_appeared per (city, scenario, episode) between the new
# FCA/FCA-Capacited CSV and the existing seed42 RL/TP/HAPC CSVs.

NEW="c:/ConflictualMAS/results/Option_OP_results/fca_vs_fca_capacited/sota_standalone_seed42.csv"
RL="c:/ConflictualMAS/results/Evaluation Files/episodes_seed42_RL.csv"
TP="c:/ConflictualMAS/results/Evaluation Files/episodes_seed42_TP.csv"
HAPC="c:/ConflictualMAS/results/Evaluation Files/episodes_seed42_HAPC.csv"

echo "=== Cross-check tasks_appeared on (city, scenario, episode) ==="
echo ""
echo "Reference cells (from existing CSVs):"
awk -F',' 'NR==1{for(i=1;i<=NF;i++) col[$i]=i; next} $col["city"]=="Tokyo_Small" && $col["phase"]=="eval_normal_wave"{print "  RL  Tokyo_Small normal_wave ep="$col["global_episode"]": tasks_appeared="$col["tasks_appeared"]" n_agents="$col["n_agents_max"]; if(++c>=2) exit}' "$RL"
awk -F',' 'NR==1{for(i=1;i<=NF;i++) col[$i]=i; next} $col["city"]=="Tokyo_Small" && $col["scenario"]=="normal_wave"{print "  HAPC Tokyo_Small normal_wave ep="$col["episode"]": tasks_appeared="$col["tasks_appeared"]" n_agents="$col["n_agents"]; if(++c>=2) exit}' "$HAPC"
echo ""
echo "NEW FCA cells (from current run):"
awk -F',' 'NR==1{for(i=1;i<=NF;i++) col[$i]=i; next} ($col["solver"]=="FaithfulCongestionAware" || $col["solver"]=="FaithfulCongestionAwareCapacited") && $col["city"]=="Tokyo_Small" && $col["scenario"]=="normal_wave"{print "  "$col["solver"]" Tokyo_Small normal_wave ep="$col["episode"]": tasks_appeared="$col["tasks_appeared"]" n_agents="$col["n_agents"]}' "$NEW"
echo ""
echo "If tasks_appeared and n_agents MATCH (=115 and 6), homogeneity is confirmed."
