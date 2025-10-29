$ErrorActionPreference = "Stop"

$segments = @(
    @{ Start = "2016-06-01"; End = "2016-12-31" },
    @{ Start = "2017-01-01"; End = "2019-12-31" },
    @{ Start = "2020-01-01"; End = "2022-12-31" },
    @{ Start = "2023-01-01"; End = "2025-10-29" }
)

foreach ($segment in $segments) {
    $start = $segment.Start
    $end = $segment.End

    Write-Host "=== Processing $start to $end ==="
    python .\gdelt_gkg_fetch_aggregate.py `
        --start $start `
        --end $end `
        --raw-dir data\gdelt_raw `
        --out-parquet data\gdelt_hourly.parquet `
        --out-csv data\gdelt_hourly.csv `
        --workers 8 `
        --only-missing `
        --chunk-hours 24

    foreach ($path in @("data\gdelt_hourly.parquet", "data\gdelt_hourly.csv")) {
        if (Test-Path $path) {
            $item = Get-Item $path
            Write-Host ("`t{0} size: {1:N0} bytes" -f $path, $item.Length)
            Write-Host ("`t{0} tail:" -f $path)
            Get-Content $path -Tail 3 | ForEach-Object { Write-Host ("`t`t{0}" -f $_) }
        } else {
            Write-Warning ("`t{0} not found after processing segment." -f $path)
        }
    }
}

if (Test-Path "data\gdelt_hourly.csv") {
    $rowCount = (Import-Csv "data\gdelt_hourly.csv").Count
    Write-Host ("Final CSV row count: {0}" -f $rowCount)
} else {
    Write-Warning "Final CSV file not found for row count summary."
}
