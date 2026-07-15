#!/usr/bin/perl
use strict; use warnings;

my $file = 'C:/ConflictualMAS/results/episodes_seed42.csv';
open my $fh, '<', $file or die $!;
my $hdr = <$fh>; $hdr =~ s/[\r\n]+$//;
my @cols = split /,/, $hdr;
my %ci; $ci{$cols[$_]} = $_ for 0..$#cols;
my @rows;
while (my $l = <$fh>) { $l =~ s/[\r\n]+$//; push @rows, [split /,/, $l]; }
close $fh;
print "Total rows: ", scalar(@rows), "\n\n";

sub get { my ($r,$c) = @_; return defined $ci{$c} ? $r->[$ci{$c}] : undef; }
sub mean { my @v = grep { defined && /\d/ } @_; return 0 unless @v; my $s=0; $s+=$_ for @v; return $s/@v; }
sub stddev { my @v = grep { defined && /\d/ } @_; return 0 if @v < 2;
             my $m = mean(@v); my $s=0; for (@v) { $s += ($_-$m)**2 } return sqrt($s/@v); }

# ── Determinism check: same (city, phase, ep_index) → identical ghost/scenario ──
print "=== DETERMINISTIC SEEDING CHECK (n_ghost_active_mean + congestion_profile per ep slot) ===\n";
# Reorganize per (city, phase, ep_slot_index)
my %eps;
for my $r (@rows) {
    my $key = get($r,'city').'|'.get($r,'phase');
    push @{$eps{$key}}, $r;
}
# For each (city, phase), check if modes share same n_ghost / congestion_profile
my $det_ok = 1; my $det_checked = 0;
for my $key (sort keys %eps) {
    my @list = @{$eps{$key}};
    # group by mode: each mode has 5 episodes (one per ep_idx)
    my %by_mode;
    push @{$by_mode{get($_, 'policy_mode')}}, $_ for @list;
    # check that for each ep_idx, n_ghost and profile match across modes
    my @modes_here = sort keys %by_mode;
    next unless @modes_here >= 2;
    my $n_eps = scalar @{$by_mode{$modes_here[0]}};
    for (my $i = 0; $i < $n_eps; ++$i) {
        my %ghosts; my %profs;
        for my $mode (@modes_here) {
            my $r = $by_mode{$mode}[$i];
            next unless defined $r;
            $ghosts{ get($r,'n_ghost_active_mean') } = 1;
            $profs{ get($r,'congestion_profile') }    = 1;
        }
        $det_checked++;
        if (keys(%ghosts) > 1 || keys(%profs) > 1) {
            $det_ok = 0;
            printf "  ! Non-deterministic: %s ep%d ghosts=[%s] profiles=[%s]\n",
                $key, $i, join(',', keys %ghosts), join(',', keys %profs);
            last;
        }
    }
}
print "  Checked $det_checked (city,phase,ep) slots — ",
    ($det_ok ? "ALL identical across modes ✓ deterministic seeding works" : "MISMATCH"), "\n\n";

# ── Per-mode aggregate ──
my @modes = sort keys %{{map { get($_,'policy_mode') => 1 } @rows}};
print "=== Per-mode aggregate (all 50 eval+stress+generalize episodes per mode) ===\n";
my @show = qw(throughput_rate accept_rate completion_per_accepted latency_mean
              mean_congestion peak_congestion mean_overlap_edges congestion_variance
              route_congestion_exposure mean_extra_steps_per_task
              agent_completed_gini max_agent_completed min_agent_completed
              total_fleet_distance_m n_ghost_active_mean wallclock_ms);
printf "%-30s", "Metric";
for my $m (@modes) { printf "%18s", $m; }
print "\n";
for my $col (@show) {
    printf "%-30s", $col;
    for my $m (@modes) {
        my @vals = map { get($_, $col) } grep { get($_,'policy_mode') eq $m } @rows;
        printf "%18.4f", mean(@vals);
    }
    print "\n";
}

# ── Per-city per-mode focus on congestion impact ──
my @cities = sort keys %{{map { get($_,'city') => 1 } @rows}};
print "\n=== Congestion impact per (city, mode) ===\n";
printf "%-20s %-18s %12s %12s %12s %12s %12s\n",
    "City","Mode","peak","overlap","cong_var","route_exp","mean_cong";
for my $city (@cities) {
    for my $mode (@modes) {
        my @r = grep { get($_,'city') eq $city && get($_,'policy_mode') eq $mode } @rows;
        next unless @r;
        printf "%-20s %-18s %12.3f %12.3f %12.4f %12.3f %12.4f\n",
            $city, $mode,
            mean(map{get($_,'peak_congestion')}@r),
            mean(map{get($_,'mean_overlap_edges')}@r),
            mean(map{get($_,'congestion_variance')}@r),
            mean(map{get($_,'route_congestion_exposure')}@r),
            mean(map{get($_,'mean_congestion')}@r);
    }
}

# ── Per-city per-mode focus on outcome (throughput, fleet distance, gini) ──
print "\n=== Outcome per (city, mode) ===\n";
printf "%-20s %-18s %10s %10s %10s %10s %10s\n",
    "City","Mode","thr","compl/acc","fleet_dist","agent_gini","wallclk_s";
for my $city (@cities) {
    for my $mode (@modes) {
        my @r = grep { get($_,'city') eq $city && get($_,'policy_mode') eq $mode } @rows;
        next unless @r;
        printf "%-20s %-18s %10.4f %10.4f %10.0f %10.4f %10.1f\n",
            $city, $mode,
            mean(map{get($_,'throughput_rate')}@r),
            mean(map{get($_,'completion_per_accepted')}@r),
            mean(map{get($_,'total_fleet_distance_m')}@r),
            mean(map{get($_,'agent_completed_gini')}@r),
            mean(map{get($_,'wallclock_ms')}@r)/1000;
    }
}

# ── Phase comparison (eval vs stress vs generalize) ──
print "\n=== Aggregate by phase ===\n";
my @phases = sort keys %{{map { get($_,'phase') => 1 } @rows}};
printf "%-12s %-18s %10s %10s %10s %10s\n",
    "Phase","Mode","thr","peak_cong","cong_var","route_exp";
for my $ph (@phases) {
    for my $m (@modes) {
        my @r = grep { get($_,'phase') eq $ph && get($_,'policy_mode') eq $m } @rows;
        next unless @r;
        printf "%-12s %-18s %10.4f %10.3f %10.4f %10.3f\n",
            $ph, $m,
            mean(map{get($_,'throughput_rate')}@r),
            mean(map{get($_,'peak_congestion')}@r),
            mean(map{get($_,'congestion_variance')}@r),
            mean(map{get($_,'route_congestion_exposure')}@r);
    }
}
