# Soft Candidates Analysis Report

**Date**: 2025-11-16 16:00 UTC+8
**Objective**: Review Ridge Composite Soft candidates and identify root causes preventing Hard IC达标
**Analyst**: Claude Code (Automated)

---

## Executive Summary

### Current Status
- **Ridge Composite (5 buckets)**: 3 Soft候選配置 (H=1/2/3, lag=1h)
- **IC Performance**: EXCELLENT (IC=0.82, far exceeds 0.02 threshold)
- **IR Performance**: EXCELLENT (IR=1.94, far exceeds 0.5 threshold)
- **PMR Performance**: **CRITICAL FAILURE** (PMR=0.0, requires >=0.55)

### Root Cause
**pos_month_ratio = 0.0** is the sole blocker preventing Hard IC达标. Despite outstanding IC and IR metrics, zero positive months means the strategy has NO months with positive IC, failing the stability requirement.

---

## Detailed Findings

### 1. Ridge Composite Performance (Task 1)

| Metric | H=1 | H=2 | H=3 | Threshold | Status |
|--------|-----|-----|-----|-----------|--------|
| IC Mean | 0.8262 | 0.8256 | 0.8161 | >=0.02 | ✓ PASS |
| IR | 1.9407 | 1.9331 | 1.8122 | >=0.5 | ✓ PASS |
| PMR | 0.0 | 0.0 | 0.0 | >=0.55 | ✗ **FAIL** |
| n_total | 2535 | 2535 | 2535 | - | - |
| month_n | 11 | 11 | 11 | - | - |

**Critical Issue**: All 11 months have IC <= 0, resulting in PMR=0.0.

**Window Analysis**:
- Total windows evaluated: 45 (15 per H)
- Windows with hard candidates found: 6 (all in最recent windows)
- All windows failed with reason: "too_few_after_lag_or_na"
- Best performing window: 2025-08-05 to 2025-10-24 (336 candidates found but rejected)

### 2. Individual Bucket Performance (Tasks 2 & 3)

#### USD_RATE Bucket
| H | IC Mean | IR | PMR | n_total | month_n |
|---|---------|-----|-----|---------|---------|
| 1 | -0.0659 | NaN | 0.0 | 468 | 1 |
| 2 | -0.0444 | NaN | 0.0 | 468 | 1 |
| 3 | -0.0685 | NaN | 0.0 | 468 | 1 |

**Diagnosis**:
- Negative IC (anti-correlated with returns)
- Only 1 month of data evaluated
- Insufficient precision in current mapping rules

#### GEOPOL Bucket
| H | IC Mean | IR | PMR | n_total | month_n |
|---|---------|-----|-----|---------|---------|
| 1 | -0.0930 | NaN | 0.0 | 468 | 1 |
| 2 | -0.0338 | NaN | 0.0 | 468 | 1 |
| 3 | -0.0217 | NaN | 0.0 | 468 | 1 |

**Diagnosis**:
- Worse negative IC than USD_RATE
- Only 1 month of data
- Current mapping rules not capturing predictive signals

### 3. Data Structure Analysis

#### GDELT Features (gdelt_hourly.parquet)
- **Shape**: 8,201 rows × 29 columns
- **Date Range**: 2024-10-29 to present
- **Buckets Available**:
  - OIL_CORE (4 columns: art_cnt, tone_avg, tone_pos_ratio, topic_cnt)
  - GEOPOL (4 columns)
  - USD_RATE (4 columns)
  - SUPPLY_CHAIN (4 columns)
  - MACRO (4 columns)
  - ESG_POLICY (4 columns)
  - ALL (aggregate, 5 columns)

#### Price Features (features_hourly.parquet)
- **Shape**: 8,757 rows × 3 columns
- **Columns**: ts_utc, ret_1h, is_trading_hour
- **Date Range**: 2024-10-29 to 2025-10-29

**Data Alignment Issue**: Price data has more rows (8,757) than GDELT (8,201), suggesting ~6.4% missing GDELT coverage.

---

## Root Cause Analysis

### Why PMR = 0.0?

**Hypothesis 1: Evaluation Pipeline Issue**
- The "too_few_after_lag_or_na" reason suggests data availability problems
- After applying lag=1h, many rows become unavailable (NaN or missing)
- This may cause month-level aggregation to fail or produce unstable ICs

**Hypothesis 2: Month-Level IC Calculation**
- If each month has too few samples after lag, monthly IC may be unreliable
- All 11 months showing IC <= 0 suggests systematic issue, not random variation
- Possible sign flip or calculation error in evaluation script

**Hypothesis 3: Data Merge Issues**
- Features (8,757 rows) vs GDELT (8,201 rows) mismatch
- Inner join may drop critical rows
- GDELT coverage gaps causing NaN propagation

### Why Individual Buckets Fail?

**USD_RATE & GEOPOL**:
1. **Only 1 month evaluated** (month_n=1) vs Ridge composite's 11 months
2. **Negative IC**: Current mapping rules capturing anti-signals
3. **Insufficient precision**: Generic keywords not specific enough to oil prices

**Precision Opportunities**:
- USD_RATE bucket needs:
  - Federal Reserve specific keywords (FOMC, Powell, interest rates)
  - Dollar index (DXY) related themes
  - Oil-denominated currency impacts

- GEOPOL bucket needs:
  - OPEC+ meeting keywords
  - Sanctions (Russia, Iran, Venezuela)
  - Middle East conflict zones (Gaza, Red Sea, Strait of Hormuz)
  - Supply disruption events

---

## Recommendations

### Priority 1: Fix PMR=0.0 Issue (CRITICAL)
1. **Investigate evaluation script** - Find IC calculation code
2. **Debug month-level aggregation** - Check why all months negative
3. **Validate data merge** - Ensure GDELT+price alignment correct
4. **Check for sign errors** - Verify IC correlation direction

### Priority 2: Increase Individual Bucket Performance
1. **USD_RATE precision v1**:
   - Add Fed-specific keywords
   - Weight by policy event importance
   - Filter out generic "dollar" mentions without Fed context

2. **GEOPOL precision v1**:
   - Add OPEC+ meeting calendar
   - Geofence to oil-producing regions
   - Event weighting by supply impact

### Priority 3: Investigate Data Coverage
1. **Fix GDELT gaps** - Why 8,201 vs 8,757 rows?
2. **Audit No-Drift compliance** - Verify mapped_ratio >= 0.55
3. **Review skip_ratio** - Ensure <= 2% per contract

---

## Next Steps (Aligned with TODO.md)

### Immediate Actions
1. ✓ **Task 1 Complete**: Soft candidates reviewed
2. ✓ **Task 2 Complete**: USD_RATE analyzed
3. ✓ **Task 3 Complete**: GEOPOL analyzed
4. **Task 4 NEXT**: Design USD_RATE enhancement strategy (v1 policy)
5. **Task 5 NEXT**: Design GEOPOL enhancement strategy (v1 policy)

### Blockers to Resolve
- **CRITICAL**: Understand why PMR=0.0 before proceeding to precision work
- **HIGH**: Locate and review IC evaluation scripts
- **MEDIUM**: Fix data alignment issues

---

## Technical Details

### Files Analyzed
```
warehouse/ic/composite5_ridge_soft_summary_short.csv (3 rows)
warehouse/ic/window_log_ridge.csv (45 rows)
warehouse/ic/ic_usd_rate_soft_summary_short.csv (3 rows)
warehouse/ic/ic_geopol_soft_summary_short.csv (3 rows)
data/gdelt_hourly.parquet (8,201 rows, 29 cols)
data/features_hourly.parquet (8,757 rows, 3 cols)
```

### Ridge Composite Details
- **Feature Name**: score_comp5_ridge
- **Buckets**: Likely USD_RATE, GEOPOL, SUPPLY_CHAIN, MACRO, ESG_POLICY (OIL_CORE excluded?)
- **Method**: Ridge regression with standardization
- **Training Windows**: 60-day rolling, 20-day test
- **Sample Size**: 2,535 total observations across 11 months

---

## Appendix: Raw Output

See: `C:\Users\niuji\Documents\Data\analysis_output.txt`

---

**Report Generated**: 2025-11-16 16:05 UTC+8
**Status**: Awaiting PMR=0.0 root cause investigation before proceeding to precision enhancements
