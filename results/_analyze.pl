#!/usr/bin/perl
use strict; use warnings;

my @files = ('C:/ConflictualMAS/results/episodes_seed42.csv',
             'C:/ConflictualMAS/results/episodes_seed43.csv');

my @rows;
my %ci;
for my $file (@files) {
    open my $fh, '<', $file or die "$file: $!";
    my $hdr = <$fh>; $hdr =~ s/[\r\n]+$//;
    my @cols = split /,/, $hdr;
    %ci = (); $ci{$cols[$_]} = $_ for 0..$#cols;
    while (my $l = <$fh>) {
        $l =~ s/[\r\n]+$//;
        my @f = split /,/, $l;
        push @rows, \@f;
    }
    close $fh;
}
my $n = scalar @rows;
print "Total rows: $n (across 2 seeds)\n";

sub get { my ($r,$c) = @_; return $r->[$ci{$c}]; }
sub mean { my @v = grep { defined && /\d/ } @_; return 0 unless @v; my $s=0; $s+=$_ for @v; return $s/@v; }
sub stddev { my @v = grep { defined && /\d/ } @_; return 0 if @v < 2;
             my $m = mean(@v); my $s=0; for (@v) { $s += ($_-$m)**2 } return sqrt($s/@v); }

# Group by (seed, mode, city)
my %g;
push @{$g{ get($_,'seed').'|'.get($_,'policy_mode').'|'.get($_,'city') }}, $_ for @rows;

print "\n=== Counts per (seed, mode, city) ===\n";
for my $k (sort keys %g) {
    printf "  %-50s %d eps\n", $k, scalar @{$g{$k}};
}

# Profile distribution per seed
print "\n=== Congestion profile distribution per seed ===\n";
for my $seed (42, 43) {
    my %p;
    for my $r (@rows) { next unless get($r,'seed') eq $seed; $p{get($r,'congestion_profile')||'(empty)'}++; }
    my $tot = 0; $tot += $_ for values %p;
    print "  seed=$seed:";
    printf "  %s=%d (%.0f%%)", $_, $p{$_}, $p{$_}/$tot*100 for sort keys %p;
    print "\n";
}

# Headline metrics: aggregate over seeds + cities per mode
my @cols_show = qw(
  throughput_rate accept_rate completion_per_accepted latency_mean
  agent_completed_gini agent_completed_std
  mean_imp_accepted mean_imp_refused
  accept_rate_high_cong accept_rate_low_cong
  mean_congestion mean_congestion_at_decision n_ghost_active_mean
  mean_extra_steps_per_task
  actor_loss entropy wallclock_ms);

print "\n=== Per-mode headline (mean ± std across 2 seeds × 3 cities × 80 rounds) ===\n";
printf "%-18s", "Metric";
printf " %14s ± %-6s", "MAPPO", "std";
printf " %14s ± %-6s", "FaithfulMAPPER", "std";
print "  ΔMAPPO−FM\n";
for my $col (@cols_show) {
    my @mappo = map { get($_, $col) } grep { get($_,'policy_mode') eq 'MAPPO' } @rows;
    my @fmap  = map { get($_, $col) } grep { get($_,'policy_mode') eq 'FaithfulMAPPER' } @rows;
    my $mm = mean(@mappo); my $sm = stddev(@mappo);
    my $mf = mean(@fmap);  my $sf = stddev(@fmap);
    my $delta = $mm - $mf;
    printf "%-18s %14.4f ± %-6.4f %14.4f ± %-6.4f  %+8.4f\n",
        $col, $mm, $sm, $mf, $sf, $delta;
}

# Convergence per mode per seed (first 25% vs last 25% of records)
print "\n=== Convergence per mode (early Q1 vs late Q4) ===\n";
for my $mode ('MAPPO','FaithfulMAPPER') {
    print "  --- $mode ---\n";
    for my $seed (42, 43) {
        for my $city ('Tokyo_Small','Kyoto_Small','LosAngeles_Small') {
            my @r = @{$g{"$seed|$mode|$city"}};
            my @early = @r[0 .. int(@r/4)-1];
            my @late  = @r[int(@r*3/4) .. $#r];
            printf "    seed=%d %-22s thr: %.3f→%.3f  acc: %.3f→%.3f  c/a: %.3f→%.3f  aloss: %.4f→%.4f  entropy: %.3f→%.3f\n",
                $seed, $city,
                mean(map{get($_,'throughput_rate')}@early), mean(map{get($_,'throughput_rate')}@late),
                mean(map{get($_,'accept_rate')}@early),     mean(map{get($_,'accept_rate')}@late),
                mean(map{get($_,'completion_per_accepted')}@early), mean(map{get($_,'completion_per_accepted')}@late),
                mean(map{get($_,'actor_loss')}@early), mean(map{get($_,'actor_loss')}@late),
                mean(map{get($_,'entropy')}@early), mean(map{get($_,'entropy')}@late);
        }
    }
}

# Selectivity: discrimination signal
print "\n=== Selectivity signal per mode (mean across 2 seeds) ===\n";
for my $mode ('MAPPO','FaithfulMAPPER') {
    my @r = grep { get($_,'policy_mode') eq $mode } @rows;
    my $ia = mean(map { get($_, 'mean_imp_accepted') } @r);
    my $ir = mean(map { get($_, 'mean_imp_refused')  } @r);
    my $ah = mean(map { get($_, 'accept_rate_high_cong') } @r);
    my $al = mean(map { get($_, 'accept_rate_low_cong')  } @r);
    printf "  %-16s imp_acc=%.3f imp_ref=%.3f  Δ_imp=%+.3f  | acc_hi_cong=%.3f acc_lo_cong=%.3f  Δ_cong=%+.3f\n",
        $mode, $ia, $ir, $ia-$ir, $ah, $al, $al-$ah;
}

# Cross-seed variance check: do the two seeds agree?
print "\n=== Cross-seed agreement (per mode, mean per seed) ===\n";
for my $mode ('MAPPO','FaithfulMAPPER') {
    for my $col (qw(throughput_rate accept_rate completion_per_accepted entropy)) {
        my @s42 = map{get($_,$col)} grep { get($_,'policy_mode') eq $mode && get($_,'seed') eq '42' } @rows;
        my @s43 = map{get($_,$col)} grep { get($_,'policy_mode') eq $mode && get($_,'seed') eq '43' } @rows;
        printf "  %-16s %-25s seed42=%.4f  seed43=%.4f  diff=%+.4f\n",
            $mode, $col, mean(@s42), mean(@s43), mean(@s42)-mean(@s43);
    }
}
