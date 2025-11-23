$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GDELT Backfill: 2025-03 to 2025-09" -ForegroundColor Cyan
Write-Host "Purpose: Fill missing bucket data gap" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Split into smaller chunks to reduce memory usage and allow progress tracking
$segments = @(
    @{ Start = "2025-03-01"; End = "2025-04-01" },
    @{ Start = "2025-04-01"; End = "2025-05-01" },
    @{ Start = "2025-05-01"; End = "2025-06-01" },
    @{ Start = "2025-06-01"; End = "2025-07-01" },
    @{ Start = "2025-07-01"; End = "2025-08-01" },
    @{ Start = "2025-08-01"; End = "2025-09-01" },
    @{ Start = "2025-09-01"; End = "2025-10-01" }
)

$totalSegments = $segments.Count
$currentSegment = 0

foreach ($segment in $segments) {
    $currentSegment++
    $start = $segment.Start
    $end = $segment.End

    Write-Host ""
    Write-Host "=== Segment $currentSegment/$totalSegments : $start to $end ===" -ForegroundColor Yellow
    Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    try {
        python .\gdelt_gkg_fetch_aggregate.py `
            --start $start `
            --end $end `
            --raw-dir data\gdelt_raw `
            --out-parquet data\gdelt_hourly.parquet `
            --out-csv data\gdelt_hourly.csv `
            --workers 8 `
            --only-missing `
            --chunk-hours 24

        $stopwatch.Stop()
        Write-Host "Segment completed in: $($stopwatch.Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green

        # Check output files
        foreach ($path in @("data\gdelt_hourly.parquet", "data\gdelt_hourly.csv")) {
            if (Test-Path $path) {
                $item = Get-Item $path
                Write-Host "`t$path size: $($item.Length / 1MB) MB" -ForegroundColor Green
            } else {
                Write-Warning "`t$path not found after processing segment."
            }
        }
    }
    catch {
        $stopwatch.Stop()
        Write-Host "ERROR in segment $start to $end : $_" -ForegroundColor Red
        Write-Host "Elapsed before error: $($stopwatch.Elapsed.ToString('hh\:mm\:ss'))"
        throw
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Backfill Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Final summary
if (Test-Path "data\gdelt_hourly.parquet") {
    $parquetSize = (Get-Item "data\gdelt_hourly.parquet").Length / 1MB
    Write-Host "Final parquet size: $parquetSize MB" -ForegroundColor Green
}

if (Test-Path "data\gdelt_hourly.csv") {
    try {
        $rowCount = (Import-Csv "data\gdelt_hourly.csv").Count
        Write-Host "Final CSV row count: $rowCount" -ForegroundColor Green
    }
    catch {
        Write-Host "Could not count CSV rows (file may be too large)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Verify monthly files in data\gdelt_hourly_monthly\" -ForegroundColor White
Write-Host "  2. Run rebuild_gdelt_total_from_monthly.py" -ForegroundColor White
Write-Host "  3. Verify 60-column structure with bucket data" -ForegroundColor White
Write-Host "  4. Run IC/PMR evaluation" -ForegroundColor White
