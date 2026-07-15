#!/usr/bin/perl
use strict; use warnings;
my $file = 'C:/ConflictualMAS/results/episodes_seed42.csv';
open my $fh, '<', $file or die $!;
my $hdr = <$fh>;
my %k;
while (my $l = <$fh>) {
    $l =~ s/[\r\n]+$//;
    my @f = split /,/, $l;
    $k{ $f[2].'|'.$f[3].'|'.$f[4] }++;
}
for my $key (sort keys %k) { printf "%-58s %d\n", $key, $k{$key}; }
print "\nTOTAL: ", scalar(map { $k{$_} } keys %k), " keys, ";
my $tot = 0; $tot += $_ for values %k; print "$tot rows\n";
