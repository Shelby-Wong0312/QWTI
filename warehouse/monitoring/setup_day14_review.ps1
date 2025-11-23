# Setup Day-14 Performance Review Scheduled Task
# Scheduled for 2025-11-26 09:00:00 UTC+8

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Day-14 Performance Review Scheduler Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$TaskName = "BaseStrategy_Day14Review"
$WorkingDir = "C:\Users\niuji\Documents\Data"
$ScriptPath = "$WorkingDir\warehouse\monitoring\day14_performance_review.py"
$PythonPath = "python"  # Assumes Python in PATH
$ScheduledDate = Get-Date "2025-11-26 09:00:00"
$LogPath = "$WorkingDir\warehouse\monitoring\day14_review_execution.log"

Write-Host "[1/5] Validating script existence..." -ForegroundColor Yellow

if (-Not (Test-Path $ScriptPath)) {
    Write-Host "  ERROR: Review script not found at: $ScriptPath" -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Script found: $ScriptPath" -ForegroundColor Green

Write-Host ""
Write-Host "[2/5] Checking for existing task..." -ForegroundColor Yellow

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask) {
    Write-Host "  [WARNING] Task '$TaskName' already exists" -ForegroundColor Yellow
    $Response = Read-Host "  Do you want to replace it? (y/n)"

    if ($Response -eq 'y') {
        Write-Host "  Unregistering existing task..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "  [OK] Existing task removed" -ForegroundColor Green
    } else {
        Write-Host "  [CANCELLED] Setup aborted by user" -ForegroundColor Yellow
        exit 0
    }
} else {
    Write-Host "  [OK] No existing task found" -ForegroundColor Green
}

Write-Host ""
Write-Host "[3/5] Creating scheduled task action..." -ForegroundColor Yellow

# Create action with output redirection to log
$ActionArgs = "$ScriptPath > `"$LogPath`" 2>&1"

$Action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument $ActionArgs `
    -WorkingDirectory $WorkingDir

Write-Host "  [OK] Action created" -ForegroundColor Green
Write-Host "    Execute: $PythonPath" -ForegroundColor Gray
Write-Host "    Arguments: $ActionArgs" -ForegroundColor Gray
Write-Host "    Working Directory: $WorkingDir" -ForegroundColor Gray

Write-Host ""
Write-Host "[4/5] Creating scheduled task trigger..." -ForegroundColor Yellow

# One-time trigger on 2025-11-26 09:00:00
$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At $ScheduledDate

Write-Host "  [OK] Trigger created" -ForegroundColor Green
Write-Host "    Scheduled for: $ScheduledDate" -ForegroundColor Gray
Write-Host "    Type: One-time execution" -ForegroundColor Gray

Write-Host ""
Write-Host "[5/5] Registering scheduled task..." -ForegroundColor Yellow

# Task settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable:$false `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# Register task
$Task = Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Day-14 Performance and Risk Control Review for base_seed202_lean7_h1 strategy"

Write-Host "  [OK] Task registered successfully" -ForegroundColor Green

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Task Details:" -ForegroundColor Cyan
Write-Host "  Name: $TaskName" -ForegroundColor White
Write-Host "  Scheduled: $ScheduledDate" -ForegroundColor White
Write-Host "  Script: $ScriptPath" -ForegroundColor White
Write-Host "  Log: $LogPath" -ForegroundColor White
Write-Host ""

Write-Host "Verification Commands:" -ForegroundColor Cyan
Write-Host "  View task: Get-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Gray
Write-Host "  Run manually: Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Gray
Write-Host "  Check log: Get-Content '$LogPath'" -ForegroundColor Gray
Write-Host "  Remove task: Unregister-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Gray
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Task will automatically run on 2025-11-26 09:00:00" -ForegroundColor White
Write-Host "  2. Review output in: $LogPath" -ForegroundColor White
Write-Host "  3. Check audit report: warehouse/monitoring/day14_audit_report.json" -ForegroundColor White
Write-Host "  4. Update RUNLOG with audit results" -ForegroundColor White
Write-Host ""

Write-Host "Manual Execution (for testing):" -ForegroundColor Yellow
Write-Host "  cd $WorkingDir" -ForegroundColor Gray
Write-Host "  python warehouse/monitoring/day14_performance_review.py" -ForegroundColor Gray
Write-Host ""
