# Periodic Performance Review Guide

## Overview

This guide outlines the periodic review schedule for Base strategy (base_seed202_lean7_h1) monitoring and risk control audits per Readme.md:1 Hard KPI specifications and Dashboard.md:1 real-time monitoring vision.

## Review Schedule

### Review Frequency

| Review Type | Schedule | Script | Next Due Date |
|------------|----------|--------|---------------|
| **Day-7 Review** | First week after activation | `day7_performance_review.py` | ✅ Completed 2025-11-19 |
| **Day-14 Review** | Second week | `day14_performance_review.py` | 📅 2025-11-26 |
| **Day-30 Review** | Monthly | `day30_performance_review.py` | 📅 2025-12-19 |
| **Quarterly Review** | Every 90 days | `quarterly_performance_review.py` | 📅 2026-02-17 |

### Review Components

Each review validates:

1. **Hard Gate Compliance** (Readme.md:12-13)
   - IC median ≥ 0.02
   - Information Ratio (IR) ≥ 0.5
   - Positive Match Rate (PMR) ≥ 0.55

2. **Drawdown Analysis**
   - Maximum drawdown tracking
   - Current drawdown status
   - Recovery analysis

3. **Alert Summary**
   - Total alerts count
   - Alert frequency trends
   - Critical alerts analysis

4. **Position Statistics**
   - Utilization percentage
   - Turnover analysis
   - Extreme position frequency

5. **Data Integrity**
   - Gap detection
   - Missing value checks
   - Execution success rate

6. **Trend Analysis** (Day-14+)
   - Performance vs previous review
   - Stability assessment
   - Regime dependency

## Day-14 Review (Scheduled 2025-11-26)

### Pre-Review Checklist

- [ ] Verify monitoring data logs exist:
  - `warehouse/monitoring/base_seed202_lean7_metrics.csv`
  - `warehouse/positions/base_seed202_lean7_positions.csv`
  - `warehouse/monitoring/hourly_execution_log.csv`
  - `warehouse/monitoring/base_seed202_lean7_alerts.csv` (if any)

- [ ] Confirm Day-7 baseline report exists:
  - `warehouse/monitoring/day7_audit_report.json`

- [ ] Ensure adequate data span:
  - Minimum: 336 hours (14 days × 24 hours)
  - Expected: ~336 observations

### Execution Steps

#### 1. Run Day-14 Performance Review

```bash
cd C:\Users\niuji\Documents\Data
python warehouse/monitoring/day14_performance_review.py
```

**Expected Output:**
- Console output with 9-step review process
- `warehouse/monitoring/day14_audit_report.json` (machine-readable)
- Health score (0-100) and overall status

#### 2. Generate Dashboard Snapshot

```bash
python warehouse/monitoring/base_dashboard.py > warehouse/monitoring/day14_dashboard_snapshot.txt
```

**Expected Output:**
- Dashboard terminal output saved to file
- Strategy card with current metrics
- Health check status
- Active alerts summary

#### 3. Review Audit Report

Open and review:
```bash
# View JSON report
type warehouse\monitoring\day14_audit_report.json

# View dashboard snapshot
type warehouse\monitoring\day14_dashboard_snapshot.txt
```

**Key Metrics to Review:**
- Hard gate status (all PASS expected)
- Health score vs Day-7 (90/100 baseline)
- IC/IR/PMR trends vs Day-7
- Drawdown status
- Stability assessment

#### 4. Update RUNLOG

Add Day-14 audit results to `RUNLOG_OPERATIONS.md`:

```markdown
## [AUDIT] Day-14 Performance and Risk Control Review

### Overview
- Review Date: [timestamp]
- Health Score: [score]/100
- Status: [status]
- Data Span: [n] days, [n] observations

### Key Findings
[Copy from audit report]

### Day-7 Comparison
[Trend analysis]

### Recommendation
[Action items]

---
```

#### 5. Decision Point: Weight Ramp

If all conditions met:
- ✅ Health score ≥ 85
- ✅ All Hard gates PASS
- ✅ No critical alerts
- ✅ Drawdown < 10%
- ✅ Stability status: STABLE
- ✅ IC/IR/PMR stable or improving vs Day-7

**Consider**: Ramp weight from 15% → 20% (per `warehouse/base_monitoring_config.json` ramp schedule)

**Update configuration:**
```json
{
  "allocation": {
    "initial_weight": 0.20,  // Increased from 0.15
    "max_weight": 0.30,
    "current_ramp_stage": "30-day"
  }
}
```

## Day-30 Review (Scheduled 2025-12-19)

### Additional Components

Beyond Day-14 checks:

1. **Feature Importance Drift**
   - Compare current vs validation feature importance
   - Check for unexpected feature shifts
   - Validate no-drift policy compliance

2. **Market Regime Analysis**
   - Performance across different volatility regimes
   - Correlation with VIX/OVX
   - Stress period performance

3. **Model Retraining Assessment**
   - Check if retraining needed (IC degradation > 20%)
   - Validate training data quality
   - Schedule retraining if required

4. **Weight Ramp Decision (60-day)**
   - Consider 20% → 25% if gates passed

## Quarterly Review (90 days)

### Comprehensive Strategy Audit

1. **Performance Attribution**
   - GDELT feature contribution analysis
   - Market factor contribution (cl1_cl2, ovx)
   - Alpha decay analysis

2. **Risk Model Validation**
   - Realized vs predicted volatility
   - Correlation structure stability
   - Tail risk assessment

3. **Operational Health**
   - Data pipeline uptime
   - Execution slippage analysis
   - Cost/benefit analysis

4. **Strategic Decisions**
   - Continue/modify/halt strategy
   - Final weight ramp to 30% (if gates passed)
   - Model upgrade considerations

## Automated Scheduling

### Windows Task Scheduler Setup

Create scheduled tasks for automated review execution:

#### Day-14 Review Task

```powershell
$TaskName = "BaseStrategy_Day14Review"
$ScriptPath = "C:\Users\niuji\Documents\Data\warehouse\monitoring\day14_performance_review.py"
$PythonPath = "python"

$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $ScriptPath
$Trigger = New-ScheduledTaskTrigger -Once -At "2025-11-26 09:00:00"

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Description "Day-14 performance review"
```

#### Recurring Monthly Review

```powershell
$TaskName = "BaseStrategy_MonthlyReview"
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddDays(30) -RepetitionInterval (New-TimeSpan -Days 30)

# Execute day30_performance_review.py monthly
```

## Alert Escalation

### Review Outcome Actions

| Health Score | Status | Action Required |
|-------------|--------|-----------------|
| 90-100 | EXCELLENT | Continue monitoring |
| 75-89 | GOOD | Monitor closely, no changes |
| 60-74 | ACCEPTABLE | Daily monitoring, prepare contingency |
| 40-59 | NEEDS_ATTENTION | Reduce weight, investigate issues |
| 0-39 | CRITICAL | Halt strategy, emergency review |

### Hard Gate Failure Protocol

If any Hard gate fails (IC < 0.02, IR < 0.5, PMR < 0.55):

1. **Immediate**: Trigger alert to operations team
2. **Within 24h**: Conduct emergency review
3. **Within 48h**: Implement remediation or halt strategy
4. **Document**: Full root cause analysis in RUNLOG

### Alert Thresholds

| Alert Level | Frequency | Action |
|------------|-----------|--------|
| WARNING | 1-5 alerts/week | Log and monitor |
| ELEVATED | 6-10 alerts/week | Investigate patterns |
| CRITICAL | >10 alerts/week or any HARD_STOP | Immediate intervention |

## Data Requirements

### Minimum Data Span

- **Day-7 Review**: 168 hours (7 days × 24h)
- **Day-14 Review**: 336 hours (14 days × 24h)
- **Day-30 Review**: 720 hours (30 days × 24h)
- **Quarterly Review**: 2160 hours (90 days × 24h)

### Data Quality Gates

All reviews require:
- ✅ No gaps > 4 hours
- ✅ Execution success rate ≥ 95%
- ✅ Missing data < 2%
- ✅ No duplicate timestamps

## Documentation

### Generated Artifacts

Each review produces:

1. **JSON Report**: `warehouse/monitoring/day{N}_audit_report.json`
   - Machine-readable full audit
   - All metrics, trends, comparisons
   - Reproducibility keys

2. **Dashboard Snapshot**: `warehouse/monitoring/day{N}_dashboard_snapshot.txt`
   - Human-readable terminal output
   - Visual representation of health
   - Quick reference for operations

3. **RUNLOG Entry**: `RUNLOG_OPERATIONS.md`
   - Executive summary
   - Key findings and recommendations
   - Decision record

### Audit Trail

Maintain complete audit chain:
```
Day-7 (2025-11-19) → Day-14 (2025-11-26) → Day-30 (2025-12-19) → Quarterly
       ↓                   ↓                    ↓                     ↓
   EXCELLENT           [TBD]                [TBD]                 [TBD]
   90/100              ?/100                ?/100                 ?/100
```

## Contact and Escalation

### Review Responsibilities

| Role | Responsibility |
|------|---------------|
| **Automated System** | Execute reviews on schedule |
| **Operations** | Review reports within 24h |
| **Risk Management** | Approve weight ramps |
| **Quant Research** | Investigate anomalies |

### Escalation Path

1. **Automated Review** → Generates report
2. **Operations** → Reviews within 24h
3. **If NEEDS_ATTENTION** → Escalate to Risk Management
4. **If CRITICAL** → Emergency quant review + halt decision

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-19 | Initial guide created |
| 1.1 | 2025-11-19 | Day-14 review scheduled |

---

**Prepared By**: Claude Code
**Document Type**: Operational Procedure
**Last Updated**: 2025-11-19 18:00 UTC+8
**Next Review**: 2025-11-26 (Day-14 execution)
