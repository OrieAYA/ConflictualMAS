#!/usr/bin/perl
use strict; use warnings;

my $file = 'C:/ConflictualMAS/results/episodes_seed42.csv';
open my $fh, '<', $file or die $!;
my $hdr = <$fh>; $hdr =~ s/[\r\n]+$//;
my @cols = split /,/, $hdr;
my %ci; $ci{$cols[$_]} = $_ for 0..$#cols;

my @rows;
while (my $l = <$fh>) {
    $l =~ s/[\r\n]+$//;
    my @f = split /,/, $l;
    push @rows, \@f;
}
close $fh;

my $n = scalar @rows;
print "Total rows: $n\n";
sub get { my ($r,$c) = @_; return $r->[$ci{$c}]; }
sub mean { my @v = grep { defined && /\d/ } @_; return 0 unless @v; my $s=0; $s+=$_ for @v; return $s/@v; }
sub stddev { my @v = grep { defined && /\d/ } @_; return 0 if @v < 2;
             my $m = mean(@v); my $s=0; for (@v) { $s += ($_-$m)**2 } return sqrt($s/@v); }

# Distribution
print "\n=== Phases x cities x modes ===\n";
my %k;
$k{ get($_,'phase').'|'.get($_,'city').'|'.get($_,'policy_mode') }++ for @rows;
for my $key (sort keys %k) { printf "  %-55s %d eps\n", $key, $k{$key}; }

# Headline per (city, mode) with mean ± std
my @cols_show = qw(
  throughput_rate accept_rate completion_per_accepted latency_mean
  agent_completed_gini agent_completed_std
  mean_imp_accepted mean_imp_refused
  accept_rate_high_cong accept_rate_low_cong
  mean_congestion_at_decision n_ghost_active_mean
  mean_extra_steps_per_task wallclock_ms);

print "\n=== Per (city, mode) means ===\n";
my %g;
push @{$g{ get($_,'city').'|'.get($_,'policy_mode') }}, $_ for @rows;
printf "%-22s %-18s", "City", "Mode";
printf " %12s", substr($_,0,12) for @cols_show;
print "\n";
for my $key (sort keys %g) {
    my ($city,$mode) = split /\|/, $key;
    printf "%-22s %-18s", $city, $mode;
    for my $col (@cols_show) {
        my $m = mean(map { get($_,$col) } @{$g{$key}});
        printf " %12.4f", $m;
    }
    print "\n";
}

# Cross-mode comparison per city
print "\n=== Cross-mode comparison per city (key metrics) ===\n";
my @cities = sort keys %{{map { get($_,'city') => 1 } @rows}};
my @modes  = sort keys %{{map { get($_,'policy_mode') => 1 } @rows}};
for my $city (@cities) {
    print "  --- $city ---\n";
    printf "    %-18s %12s %12s %12s %12s %12s %12s\n",
        "Mode","throughput","accept","completion","latency","gini","extra_steps";
    for my $mode (@modes) {
        my @r = @{$g{"$city|$mode"} // []};
        next unless @r;
        printf "    %-18s %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f\n",
            $mode,
            mean(map{get($_,'throughput_rate')}@r),
            mean(map{get($_,'accept_rate')}@r),
            mean(map{get($_,'completion_per_accepted')}@r),
            mean(map{get($_,'latency_mean')}@r),
            mean(map{get($_,'agent_completed_gini')}@r),
            mean(map{get($_,'mean_extra_steps_per_task')}@r);
    }
}

# Aggregate per mode (across all eval scenarios)
print "\n=== Per mode AGGREGATE (across all cities, both phases) ===\n";
printf "%-18s %12s %12s %12s %12s %12s %12s %12s %12s\n",
    "Mode","throughput","±σ","accept","completion","latency","gini","cong_dec","wallclock";
for my $mode (@modes) {
    my @r = grep { get($_,'policy_mode') eq $mode } @rows;
    next unless @r;
    my $thr = mean(map{get($_,'throughput_rate')}@r);
    my $thr_s = stddev(map{get($_,'throughput_rate')}@r);
    printf "%-18s %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.0f\n",
        $mode, $thr, $thr_s,
        mean(map{get($_,'accept_rate')}@r),
        mean(map{get($_,'completion_per_accepted')}@r),
        mean(map{get($_,'latency_mean')}@r),
        mean(map{get($_,'agent_completed_gini')}@r),
        mean(map{get($_,'mean_congestion_at_decision')}@r),
        mean(map{get($_,'wallclock_ms')}@r);
}

# Selectivity gap analysis
print "\n=== Selectivity signal per mode ===\n";
for my $mode (@modes) {
    my @r = grep { get($_,'policy_mode') eq $mode } @rows;
    next unless @r;
    my $ia = mean(map{get($_,'mean_imp_accepted')}@r);
    my $ir = mean(map{get($_,'mean_imp_refused')}@r);
    my $ah = mean(map{get($_,'accept_rate_high_cong')}@r);
    my $al = mean(map{get($_,'accept_rate_low_cong')}@r);
    printf "  %-18s imp_acc=%.3f imp_ref=%.3f  Δ_imp=%+.3f  | acc_hi=%.3f acc_lo=%.3f  Δ_cong=%+.3f\n",
        $mode, $ia, $ir, $ia-$ir, $ah, $al, $al-$ah;
}
