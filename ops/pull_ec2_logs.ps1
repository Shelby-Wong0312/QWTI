<#
.SYNOPSIS
    Pull daily logs from AWS EC2 to local backup directory.

.DESCRIPTION
    This script pulls monitoring logs from the EC2 instance (wti-aws) to
    the local cloud_logs directory, organized by date.

    Files pulled:
    - hourly_runlog.jsonl
    - base_seed202_lean7_metrics.csv
    - base_seed202_lean7_positions.csv
    - daily_experiment_log/YYYY-MM-DD.md

.NOTES
    Schedule this script with Windows Task Scheduler to run daily at 21:00.

    Author: Data Pipeline Ops
    Date: 2025-11-23
#>

param(
    [string]$Date = (Get-Date).ToString("yyyy-MM-dd"),
    [string]$EC2Host = "wti-aws",
    [string]$RemoteBase = "/home/ec2-user/Data",
    [string]$LocalBase = "C:\Users\niuji\Documents\Data\cloud_logs"
)

# Configuration
$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Host "============================================================"
Write-Host "EC2 LOG PULL - $timestamp"
Write-Host "============================================================"
Write-Host ""

# Create date-specific local directory
$localDateDir = Join-Path $LocalBase $Date
if (-not (Test-Path $localDateDir)) {
    New-Item -ItemType Directory -Path $localDateDir -Force | Out-Null
    Write-Host "[INFO] Created directory: $localDateDir"
}

# Define files to pull
$filesToPull = @(
    @{
        Remote = "$RemoteBase/warehouse/monitoring/hourly_runlog.jsonl"
        Local = "$localDateDir/hourly_runlog.jsonl"
        Description = "Hourly Runlog (JSONL)"
    },
    @{
        Remote = "$RemoteBase/warehouse/monitoring/base_seed202_lean7_metrics.csv"
        Local = "$localDateDir/base_seed202_lean7_metrics.csv"
        Description = "Metrics Log"
    },
    @{
        Remote = "$RemoteBase/warehouse/positions/base_seed202_lean7_positions.csv"
        Local = "$localDateDir/base_seed202_lean7_positions.csv"
        Description = "Positions Log"
    },
    @{
        Remote = "$RemoteBase/warehouse/monitoring/base_seed202_lean7_alerts.csv"
        Local = "$localDateDir/base_seed202_lean7_alerts.csv"
        Description = "Alerts Log"
    },
    @{
        Remote = "$RemoteBase/warehouse/monitoring/daily_experiment_log/$Date.md"
        Local = "$localDateDir/daily_experiment_log_$Date.md"
        Description = "Daily Experiment Log"
    },
    @{
        Remote = "$RemoteBase/warehouse/monitoring/hourly_execution_log.csv"
        Local = "$localDateDir/hourly_execution_log.csv"
        Description = "Execution Log"
    }
)

# Pull each file
$successCount = 0
$failCount = 0

foreach ($file in $filesToPull) {
    Write-Host ""
    Write-Host "[PULL] $($file.Description)"
    Write-Host "       Remote: $($file.Remote)"
    Write-Host "       Local:  $($file.Local)"

    try {
        # Use scp to pull the file
        $scpResult = scp "${EC2Host}:$($file.Remote)" "$($file.Local)" 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Host "       Status: SUCCESS" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host "       Status: FAILED (file may not exist)" -ForegroundColor Yellow
            $failCount++
        }
    }
    catch {
        Write-Host "       Status: ERROR - $($_.Exception.Message)" -ForegroundColor Red
        $failCount++
    }
}

# Summary
Write-Host ""
Write-Host "============================================================"
Write-Host "PULL COMPLETE"
Write-Host "============================================================"
Write-Host "Date:      $Date"
Write-Host "Local Dir: $localDateDir"
Write-Host "Success:   $successCount"
Write-Host "Failed:    $failCount"
Write-Host ""

# Create a manifest file
$manifestPath = Join-Path $localDateDir "manifest.json"
$manifest = @{
    pull_timestamp = $timestamp
    date = $Date
    ec2_host = $EC2Host
    remote_base = $RemoteBase
    local_dir = $localDateDir
    files_attempted = $filesToPull.Count
    files_success = $successCount
    files_failed = $failCount
} | ConvertTo-Json -Depth 3

$manifest | Out-File -FilePath $manifestPath -Encoding utf8
Write-Host "[INFO] Manifest written to: $manifestPath"

# Exit with appropriate code
if ($failCount -gt 0 -and $successCount -eq 0) {
    Write-Host "[WARN] All pulls failed - check EC2 connectivity" -ForegroundColor Red
    exit 1
} else {
    exit 0
}
