# compare_baselines.ps1 - aggregate MAPPO vs baselines results into a markdown report.
# Covers the objective functions described in the research PDF (slides 17 + 44):
#   Obj 1  Throughput               (tasks_completed / tasks_appeared)
#   Obj 2  Completeness latency     (steps pickup -> delivery)
#   Obj 3  Task allocation accept_rate
#   Obj 4  Computation cost ratio   (wallclock_ms vs Greedy reference)
#   Bonus  Scalability across cities
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\compare_baselines.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\compare_baselines.ps1 -Phase generalize
#
# Reads non-empty C:\ConflictualMAS\results\episodes_seed*.csv files,
# filters by `phase` (default "eval"), and writes
# C:\ConflictualMAS\results\comparison_report.md

[CmdletBinding()]
param(
    [string] $CsvPath = "",
    [string] $Phase   = "eval",
    [string] $OutPath = "C:\ConflictualMAS\results\comparison_report.md",
    [string] $RefMode = "Greedy"
)

$ErrorActionPreference = "Stop"
$resultsDir = "C:\ConflictualMAS\results"

# --- Collect input files -----------------------------------------------------
$files = @()
if ($CsvPath -ne "") {
    if (-not (Test-Path $CsvPath)) { throw "CSV not found: $CsvPath" }
    $files += $CsvPath
} else {
    Get-ChildItem "$resultsDir\episodes_seed*.csv" -ErrorAction SilentlyContinue |
        Where-Object { $_.Length -gt 0 } |
        ForEach-Object { $files += $_.FullName }
}
if ($files.Count -eq 0) { throw "No non-empty episode CSVs found in $resultsDir" }
Write-Host ("Reading {0} CSV file(s)" -f $files.Count)
$files | ForEach-Object { Write-Host ("  " + $_) }

# --- Load + filter -----------------------------------------------------------
$rows = @()
foreach ($f in $files) { $rows += Import-Csv $f }
$rows = $rows | Where-Object { $_.phase -eq $Phase }
Write-Host ("After filter phase={0}: {1} episodes" -f $Phase, $rows.Count)
if ($rows.Count -eq 0) { throw "No rows match phase=$Phase" }

# --- Stats helpers -----------------------------------------------------------
function Get-Stats {
    param([object[]]$Data, [string]$Field)
    $vals = New-Object System.Collections.ArrayList
    foreach ($d in $Data) { [void]$vals.Add([double]$d.$Field) }
    $n = $vals.Count
    if ($n -eq 0) { return [pscustomobject]@{ Mean = 0.0; Std = 0.0; N = 0 } }
    $sum = 0.0
    foreach ($v in $vals) { $sum += $v }
    $mean = $sum / $n
    $sq = 0.0
    foreach ($v in $vals) { $sq += [math]::Pow($v - $mean, 2) }
    $std = 0.0
    if ($n -gt 1) { $std = [math]::Sqrt($sq / ($n - 1)) }
    [pscustomobject]@{ Mean = $mean; Std = $std; N = $n }
}

function Format-MeanStd {
    param($Stats, [int]$Digits = 4)
    $m = [math]::Round($Stats.Mean, $Digits)
    $s = [math]::Round($Stats.Std,  $Digits)
    return ("{0} +- {1}" -f $m, $s)
}

function Repeat-Str {
    param([string]$Str, [int]$Count)
    $sb = New-Object System.Text.StringBuilder
    for ($i = 0; $i -lt $Count; $i++) { [void]$sb.Append($Str) }
    return $sb.ToString()
}

# --- Aggregate (city, policy_mode) -------------------------------------------
$groups = $rows | Group-Object city, policy_mode
$agg = @()
foreach ($g in $groups) {
    $parts = $g.Name -split ', '
    $entry = [pscustomobject]@{
        city          = $parts[0]
        policy_mode   = $parts[1]
        n_episodes    = $g.Count
        throughput    = (Get-Stats $g.Group "throughput_rate")
        accept        = (Get-Stats $g.Group "accept_rate")
        latency_mean  = (Get-Stats $g.Group "latency_mean")
        latency_pa    = (Get-Stats $g.Group "latency_per_agent")
        utilisation   = (Get-Stats $g.Group "agent_utilisation")
        wallclock     = (Get-Stats $g.Group "wallclock_ms")
    }
    $agg += $entry
}

# --- Stable orderings --------------------------------------------------------
$citiesSorted = @($agg | Select-Object -ExpandProperty city -Unique | Sort-Object)
$modeOrder    = @("MAPPO","TamAlwaysAccept","LaCAM","PIBT","CongestionAware","InsertionGreedy","Greedy","Random")
$modesPresent = @($agg | Select-Object -ExpandProperty policy_mode -Unique)
$modesSorted  = @()
foreach ($m in $modeOrder) {
    if ($modesPresent -contains $m) { $modesSorted += $m }
}

# --- Table builders ----------------------------------------------------------
function Build-PivotTable {
    param(
        [object[]]$Agg,
        [string]  $Field,
        [int]     $Digits = 4,
        [string[]]$Cities,
        [string[]]$Modes
    )
    $lines = @()
    $hdr = "| Method"
    foreach ($c in $Cities) { $hdr += " | " + $c }
    $hdr += " |"
    $lines += $hdr

    $sep = "|---"
    for ($i = 0; $i -lt $Cities.Count; $i++) { $sep += "|---" }
    $sep += "|"
    $lines += $sep

    foreach ($mode in $Modes) {
        $line = "| " + $mode
        foreach ($city in $Cities) {
            $row = $Agg | Where-Object { $_.city -eq $city -and $_.policy_mode -eq $mode } | Select-Object -First 1
            if ($null -ne $row) {
                $stats = $row.$Field
                $line += " | " + (Format-MeanStd $stats $Digits)
            } else {
                $line += " | --"
            }
        }
        $line += " |"
        $lines += $line
    }
    return ($lines -join "`n")
}

function Build-CostRatio {
    param(
        [object[]]$Agg,
        [string]  $RefMode,
        [string[]]$Cities,
        [string[]]$Modes
    )
    $lines = @()
    $hdr = "| Method"
    foreach ($c in $Cities) { $hdr += " | " + $c }
    $hdr += " |"
    $lines += $hdr
    $sep = "|---"
    for ($i = 0; $i -lt $Cities.Count; $i++) { $sep += "|---" }
    $sep += "|"
    $lines += $sep

    foreach ($mode in $Modes) {
        $line = "| " + $mode
        foreach ($city in $Cities) {
            $row = $Agg | Where-Object { $_.city -eq $city -and $_.policy_mode -eq $mode } | Select-Object -First 1
            $ref = $Agg | Where-Object { $_.city -eq $city -and $_.policy_mode -eq $RefMode } | Select-Object -First 1
            if ($null -ne $row -and $null -ne $ref -and $ref.wallclock.Mean -gt 0) {
                $r = [math]::Round($row.wallclock.Mean / $ref.wallclock.Mean, 2)
                $line += " | " + $r + "x"
            } else {
                $line += " | --"
            }
        }
        $line += " |"
        $lines += $line
    }
    return ($lines -join "`n")
}

function Build-CountTable {
    param(
        [object[]]$Agg,
        [string[]]$Cities,
        [string[]]$Modes
    )
    $lines = @()
    $hdr = "| Method"
    foreach ($c in $Cities) { $hdr += " | " + $c }
    $hdr += " |"
    $lines += $hdr
    $sep = "|---"
    for ($i = 0; $i -lt $Cities.Count; $i++) { $sep += "|---" }
    $sep += "|"
    $lines += $sep

    foreach ($mode in $Modes) {
        $line = "| " + $mode
        foreach ($city in $Cities) {
            $row = $Agg | Where-Object { $_.city -eq $city -and $_.policy_mode -eq $mode } | Select-Object -First 1
            if ($null -ne $row) {
                $line += " | " + $row.n_episodes
            } else {
                $line += " | 0"
            }
        }
        $line += " |"
        $lines += $line
    }
    return ($lines -join "`n")
}

# --- Build the report --------------------------------------------------------
$dateStr = Get-Date -Format "yyyy-MM-dd HH:mm"

$lines = New-Object System.Collections.ArrayList
[void]$lines.Add("# Comparison Report - phase = ``$Phase``")
[void]$lines.Add("")
[void]$lines.Add("_Generated $dateStr from $($files.Count) seed CSV(s)._")
[void]$lines.Add("_Mapping to objective functions: research PDF slides 17, 44._")
[void]$lines.Add("")

[void]$lines.Add("## Obj. 1 - Throughput (tasks_completed / tasks_appeared)")
[void]$lines.Add("")
[void]$lines.Add("Higher is better. Mean +- std across eval episodes.")
[void]$lines.Add("")
[void]$lines.Add((Build-PivotTable -Agg $agg -Field "throughput" -Digits 4 -Cities $citiesSorted -Modes $modesSorted))
[void]$lines.Add("")

[void]$lines.Add("## Obj. 2 - Completeness latency (steps, pickup -> delivery)")
[void]$lines.Add("")
[void]$lines.Add("Lower is better. ``latency_mean`` averaged across delivered tasks.")
[void]$lines.Add("")
[void]$lines.Add((Build-PivotTable -Agg $agg -Field "latency_mean" -Digits 1 -Cities $citiesSorted -Modes $modesSorted))
[void]$lines.Add("")
[void]$lines.Add("### Obj. 2b - Latency per available agent")
[void]$lines.Add("")
[void]$lines.Add("``latency_mean / mean_active_agents`` (slide 44, ``Completeness latency mean related to the number of available agents``).")
[void]$lines.Add("")
[void]$lines.Add((Build-PivotTable -Agg $agg -Field "latency_pa" -Digits 2 -Cities $citiesSorted -Modes $modesSorted))
[void]$lines.Add("")

[void]$lines.Add("## Obj. 3 - Task allocation: acceptance rate")
[void]$lines.Add("")
[void]$lines.Add("Higher is not always better. Under capacity constraint, over-acceptance leads to unfinished tasks.")
[void]$lines.Add("MAPPO is expected to find a task-aware sweet spot rather than max or min.")
[void]$lines.Add("")
[void]$lines.Add((Build-PivotTable -Agg $agg -Field "accept" -Digits 4 -Cities $citiesSorted -Modes $modesSorted))
[void]$lines.Add("")

[void]$lines.Add("## Obj. 4 - Computation cost ratio (wallclock vs ``$RefMode``)")
[void]$lines.Add("")
[void]$lines.Add("Lower is better. Captures the ``computation cost ratio`` requirement on slide 17.")
[void]$lines.Add("")
[void]$lines.Add((Build-CostRatio -Agg $agg -RefMode $RefMode -Cities $citiesSorted -Modes $modesSorted))
[void]$lines.Add("")

[void]$lines.Add("## Bonus - Scalability (throughput across city sizes)")
[void]$lines.Add("")
[void]$lines.Add("Tokyo_Small -> Tokyo_Medium (-> Tokyo_Large if available).")
[void]$lines.Add("Shows degradation curve as graph size grows.")
[void]$lines.Add("")
[void]$lines.Add((Build-PivotTable -Agg $agg -Field "throughput" -Digits 4 -Cities $citiesSorted -Modes $modesSorted))
[void]$lines.Add("")

[void]$lines.Add("## Auxiliary - Agent utilisation")
[void]$lines.Add("")
[void]$lines.Add("Fraction of steps with active agents. Helps disambiguate low-throughput cases (slack vs overload).")
[void]$lines.Add("")
[void]$lines.Add((Build-PivotTable -Agg $agg -Field "utilisation" -Digits 4 -Cities $citiesSorted -Modes $modesSorted))
[void]$lines.Add("")

[void]$lines.Add("## Methodology - episode counts")
[void]$lines.Add("")
[void]$lines.Add((Build-CountTable -Agg $agg -Cities $citiesSorted -Modes $modesSorted))
[void]$lines.Add("")

($lines -join "`n") | Out-File -FilePath $OutPath -Encoding utf8

Write-Host ""
Write-Host ("Report written to {0}" -f $OutPath)
Write-Host ("Cities found:  {0}" -f ($citiesSorted -join ", "))
Write-Host ("Methods found: {0}" -f ($modesSorted -join ", "))
