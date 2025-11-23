# Hourly Monitoring Scheduler Setup

This document explains how to set up automated hourly monitoring for the Base Strategy.

---

## Overview

**Purpose**: Run `hourly_monitor.py` every hour to:
1. Collect latest predictions and features
2. Calculate positions based on predictions
3. Compute rolling metrics (IC, IR, PMR)
4. Check Hard gates (Readme.md:12-13)
5. Write to `positions/` and `monitoring/` logs
6. Generate alerts if Hard gates fail
7. Support Dashboard.md terminal vision

**Schedule**: Every hour, 24/7
**Logs**:
- Positions: `warehouse/positions/base_seed202_lean7_positions.csv`
- Metrics: `warehouse/monitoring/base_seed202_lean7_metrics.csv`
- Alerts: `warehouse/monitoring/base_seed202_lean7_alerts.csv`
- Execution: `warehouse/monitoring/hourly_execution_log.csv`

---

## Method 1: Windows Task Scheduler (Recommended for Windows)

### Automated Setup (PowerShell)

1. **Open PowerShell as Administrator**

2. **Navigate to monitoring directory**:
   ```powershell
   cd C:\Users\niuji\Documents\Data\warehouse\monitoring
   ```

3. **Run setup script**:
   ```powershell
   .\setup_hourly_scheduler.ps1
   ```

4. **Verify task was created**:
   ```powershell
   Get-ScheduledTask -TaskName "BaseStrategy_HourlyMonitor"
   ```

### Manual Setup (GUI)

1. Open **Task Scheduler** (taskschd.msc)

2. Click **Create Task** (not Basic Task)

3. **General tab**:
   - Name: `BaseStrategy_HourlyMonitor`
   - Description: `Hourly monitoring for Base Strategy - maintains Hard gates`
   - Run whether user is logged on or not: ✓
   - Run with highest privileges: ✓

4. **Triggers tab**:
   - New Trigger
   - Begin: On a schedule
   - Settings: Daily, start today at current hour
   - Advanced: Repeat task every 1 hour, for a duration of Indefinitely
   - OK

5. **Actions tab**:
   - New Action
   - Program: `C:\Users\niuji\Documents\Data\.venv\Scripts\python.exe`
   - Arguments: `warehouse\monitoring\hourly_monitor.py`
   - Start in: `C:\Users\niuji\Documents\Data`
   - OK

6. **Conditions tab**:
   - Uncheck "Start only if on AC power"
   - Check "Wake the computer to run this task"

7. **Settings tab**:
   - Allow task to run on demand: ✓
   - Run task as soon as possible if missed: ✓
   - If task fails, restart every: 5 minutes, 3 attempts

8. Click **OK**

### Testing the Task

**Run immediately** (PowerShell):
```powershell
Start-ScheduledTask -TaskName "BaseStrategy_HourlyMonitor"
```

**Check last run status**:
```powershell
Get-ScheduledTaskInfo -TaskName "BaseStrategy_HourlyMonitor"
```

**View execution log**:
```powershell
cat C:\Users\niuji\Documents\Data\warehouse\monitoring\hourly_execution_log.csv
```

---

## Method 2: Python Scheduler (Alternative)

If you prefer to keep everything in Python, use `schedule` library:

### Install schedule
```bash
pip install schedule
```

### Create daemon script

Create `warehouse/monitoring/scheduler_daemon.py`:
```python
import schedule
import time
import subprocess
from pathlib import Path

def run_hourly_monitor():
    script = Path('warehouse/monitoring/hourly_monitor.py')
    subprocess.run(['python', str(script)], check=True)

# Schedule every hour
schedule.every().hour.do(run_hourly_monitor)

print("Scheduler started - running every hour")
print("Press Ctrl+C to stop")

while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
```

### Run as background service
```bash
# Windows (run in background)
pythonw warehouse\monitoring\scheduler_daemon.py

# Or use nssm (Non-Sucking Service Manager) to create Windows service
```

---

## Method 3: Cron (Linux/Mac)

If running on Linux/Mac, use cron:

```bash
# Edit crontab
crontab -e

# Add this line (runs every hour)
0 * * * * cd /path/to/Data && /path/to/.venv/bin/python warehouse/monitoring/hourly_monitor.py >> logs/hourly_monitor.log 2>&1
```

---

## Monitoring the Scheduler

### Check if running
```powershell
# Windows Task Scheduler
Get-ScheduledTask -TaskName "BaseStrategy_HourlyMonitor" | Select-Object State, LastRunTime, NextRunTime
```

### View logs
```bash
# Execution log
cat warehouse/monitoring/hourly_execution_log.csv

# Position log
cat warehouse/positions/base_seed202_lean7_positions.csv

# Metrics log
cat warehouse/monitoring/base_seed202_lean7_metrics.csv

# Alert log (only if alerts triggered)
cat warehouse/monitoring/base_seed202_lean7_alerts.csv
```

### Dashboard view
```bash
# Run dashboard to see current status
python warehouse/monitoring/base_dashboard.py
```

---

## Hard Gate Monitoring

The scheduler automatically checks Hard gates every hour:

### Data Quality Gates (Readme.md:12)
- mapped_ratio ≥ 0.55
- ALL_art_cnt ≥ 3
- tone_avg non-empty
- skip_ratio ≤ 2%

### IC Performance Gates (Readme.md:13)
- IC median ≥ 0.02 (15-day rolling)
- IR ≥ 0.5 (60-day rolling)
- PMR ≥ 0.55 (30-day rolling)

### Hard Stops (Auto De-activation)
- IC < 0.01 for 5 consecutive windows → AUTO_DEACTIVATE
- IR < 0.3 for 10 consecutive windows → AUTO_DEACTIVATE

**Alerts** are logged to `warehouse/monitoring/base_seed202_lean7_alerts.csv` when gates fail.

---

## Troubleshooting

### Task not running
1. Check task status:
   ```powershell
   Get-ScheduledTaskInfo -TaskName "BaseStrategy_HourlyMonitor"
   ```

2. Check execution log for errors:
   ```bash
   tail warehouse/monitoring/hourly_execution_log.csv
   ```

3. Run manually to see errors:
   ```bash
   python warehouse/monitoring/hourly_monitor.py
   ```

### Python not found
Update Python path in task:
```powershell
# Find Python path
where python

# Update task with correct path
```

### Permissions issues
- Run Task Scheduler as Administrator
- Ensure write permissions to warehouse/ directories

### Missing dependencies
```bash
# Activate venv and install
.venv\Scripts\activate
pip install pandas numpy lightgbm
```

---

## Disabling/Removing

### Disable (pause monitoring)
```powershell
Disable-ScheduledTask -TaskName "BaseStrategy_HourlyMonitor"
```

### Enable (resume monitoring)
```powershell
Enable-ScheduledTask -TaskName "BaseStrategy_HourlyMonitor"
```

### Remove completely
```powershell
Unregister-ScheduledTask -TaskName "BaseStrategy_HourlyMonitor" -Confirm:$false
```

---

## Production Deployment Checklist

- [ ] Python environment activated
- [ ] All dependencies installed (pandas, numpy, lightgbm)
- [ ] `warehouse/base_monitoring_config.json` verified
- [ ] `features_hourly_with_term.parquet` data available
- [ ] Write permissions to warehouse/positions/ and warehouse/monitoring/
- [ ] Task scheduler configured and running
- [ ] First hourly execution completed successfully
- [ ] Logs being written (positions, metrics, execution)
- [ ] Dashboard can read and display metrics
- [ ] Alert notifications configured (email/SMS if desired)

---

## Support & Maintenance

**Logs Location**: `warehouse/monitoring/`
**Config**: `warehouse/base_monitoring_config.json`
**Documentation**: This file + `warehouse/BASE_PROMOTION_SUMMARY.md`
**Contact**: Refer to RUNLOG_OPERATIONS.md for support escalation

**Last Updated**: 2025-11-19
