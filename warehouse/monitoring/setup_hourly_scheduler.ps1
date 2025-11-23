# PowerShell Script: Setup Windows Task Scheduler for Hourly Monitoring
#
# This script creates a scheduled task that runs hourly_monitor.py every hour
# to maintain Hard gates and support Dashboard terminal vision.
#
# Usage: Run as Administrator in PowerShell
#   .\setup_hourly_scheduler.ps1

$TaskName = "BaseStrategy_HourlyMonitor"
$ScriptPath = "C:\Users\niuji\Documents\Data\warehouse\monitoring\hourly_monitor.py"
$PythonPath = "C:\Users\niuji\Documents\Data\.venv\Scripts\python.exe"  # Adjust if needed
$WorkingDir = "C:\Users\niuji\Documents\Data"

# Check if task already exists
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask) {
    Write-Host "Task '$TaskName' already exists. Removing..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Write-Host "Creating scheduled task: $TaskName" -ForegroundColor Green

# Create action to run Python script
$Action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument $ScriptPath `
    -WorkingDirectory $WorkingDir

# Create trigger for hourly execution
$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration ([TimeSpan]::MaxValue)

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -MultipleInstances Parallel

# Register the task
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Hourly monitoring for Base Strategy (Seed202 LEAN 7-Feature). Maintains Hard gates and supports Dashboard terminal vision." `
    -User $env:USERNAME

Write-Host ""
Write-Host "Scheduled task created successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Task Details:" -ForegroundColor Cyan
Write-Host "  Name: $TaskName"
Write-Host "  Script: $ScriptPath"
Write-Host "  Python: $PythonPath"
Write-Host "  Schedule: Every hour"
Write-Host "  Working Directory: $WorkingDir"
Write-Host ""
Write-Host "To verify the task:" -ForegroundColor Yellow
Write-Host "  Get-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "To run manually:" -ForegroundColor Yellow
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "To view task history:" -ForegroundColor Yellow
Write-Host "  Get-ScheduledTaskInfo -TaskName '$TaskName'"
Write-Host ""
Write-Host "To disable the task:" -ForegroundColor Yellow
Write-Host "  Disable-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "To remove the task:" -ForegroundColor Yellow
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
