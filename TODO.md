# WTI × GDELT - TODO List

**目標**: 達成第一個 Hard IC 達標訊號 (H≤3, lag=1h; IC≥0.02 ∧ IR≥0.5 ∧ PMR≥0.55)

**最後更新**: 2025-11-16 01:34 UTC+8

---

## 階段 1: 現況分析

- [ ] **Task 1**: Review current Soft candidates (Ridge composite) to understand what's working
  - 檔案位置: `warehouse\ic\alpha_candidates_composite5_ridge_soft_short.csv`
  - 預期輸出: 分析報告 → `warehouse\analysis\soft_candidates_analysis_YYYYMMDD.md`

- [ ] **Task 2**: Analyze USD_RATE bucket precision opportunities
  - 檢視當前 mapping rules, tone distribution, coverage
  - 預期輸出: `warehouse\analysis\usd_rate_precision_analysis_YYYYMMDD.md`

- [ ] **Task 3**: Analyze GEOPOL bucket precision opportunities
  - 檢視當前 mapping rules, tone distribution, coverage
  - 預期輸出: `warehouse\analysis\geopol_precision_analysis_YYYYMMDD.md`

---

## 階段 2: 精準化策略設計

- [ ] **Task 4**: Design USD_RATE enhancement strategy
  - 關鍵字優化、事件權重調整、tone filtering
  - 預期輸出: `warehouse\policy\usd_rate_precision_v1.yaml`

- [ ] **Task 5**: Design GEOPOL enhancement strategy
  - 關鍵字優化、事件權重調整、tone filtering
  - 預期輸出: `warehouse\policy\geopol_precision_v1.yaml`

---

## 階段 3: 實作與測試

- [ ] **Task 6**: Implement USD_RATE precision v1
  - 修改 bucket mapping logic
  - 通過 No-Drift preflight 驗證
  - 預期輸出: 更新 `gdelt_gkg_fetch_aggregate.py` 或新增 precision module

- [ ] **Task 7**: Run IC evaluation for USD_RATE v1
  - 參數: H=1/2/3, lag=1h, Hard threshold
  - 預期輸出: `warehouse\ic\ic_usd_rate_v1_hard_short.csv`

- [ ] **Task 8**: Implement GEOPOL precision v1
  - 修改 bucket mapping logic
  - 通過 No-Drift preflight 驗證
  - 預期輸出: 更新 `gdelt_gkg_fetch_aggregate.py` 或新增 precision module

- [ ] **Task 9**: Run IC evaluation for GEOPOL v1
  - 參數: H=1/2/3, lag=1h, Hard threshold
  - 預期輸出: `warehouse\ic\ic_geopol_v1_hard_short.csv`

---

## 階段 4: 進階嘗試 (if Hard IC not reached)

- [ ] **Task 10**: Try composite strategy combining enhanced USD_RATE + GEOPOL
  - 使用 Ridge/Lasso/ElasticNet
  - 預期輸出: `warehouse\ic\alpha_candidates_composite_enhanced_*.csv`

- [ ] **Task 11**: Run IC evaluation for enhanced composite
  - 參數: H=1/2/3, lag=1h, Hard threshold
  - 預期輸出: Hard IC summary

- [ ] **Task 12**: Explore alternative lag values (lag=2h, 3h) as fallback
  - 僅在 lag=1h 失敗後執行
  - 預期輸出: `warehouse\ic\ic_*_lag2h_*.csv`

- [ ] **Task 13**: Investigate other bucket combinations
  - OIL + USD_RATE, OIL + GEOPOL, etc.
  - 預期輸出: `warehouse\ic\ic_*_multi_bucket_*.csv`

---

## 階段 5: Hard IC 達標後部署

- [ ] **Task 14**: Validate first Hard IC candidate against No-Drift contract
  - 執行完整 preflight check
  - 確認 Gate: mapped_ratio≥0.55, ALL_art_cnt≥3, tone_avg 非空, skip_ratio≤2%
  - 預期輸出: `warehouse\validation\hard_ic_validation_YYYYMMDD.json`

- [ ] **Task 15**: Document Hard IC parameters and rationale in runlog
  - 記錄: policy version, H, lag, bucket config, IC/IR/PMR values
  - 預期輸出: 更新 `RUNLOG_OPERATIONS.md`

- [ ] **Task 16**: Set selected_source=base for Hard IC candidate
  - 更新 config/policy file
  - 預期輸出: `warehouse\policy\selected_source.yaml`

---

## 階段 6: 生產化與監控

- [ ] **Task 17**: Implement weight generation pipeline (w_t) for Base candidate
  - 建立 hourly weight calculator
  - 預期輸出: `weight_generator.py` + `data\weights_hourly.parquet`

- [ ] **Task 18**: Build monitoring dashboard for real-time w_t tracking
  - 即時監控 w_t, IC drift, Gate violations
  - 預期輸出: `dashboard\realtime_monitor.html` or notebook

- [ ] **Task 19**: Set up alerting for Gate/IC threshold violations
  - Email/Slack notifications
  - 預期輸出: `monitoring\alert_config.yaml` + alert script

- [ ] **Task 20**: Create audit trail and replay mechanism for w_t history
  - 可回放任意時間點的 w_t 計算
  - 預期輸出: `audit\replay_engine.py`

- [ ] **Task 21**: Document final system architecture and operational procedures
  - 完整 SOP: 日常監控、異常處理、回測流程
  - 預期輸出: `docs\OPERATIONS_MANUAL.md`

---

## 完成條件 (DoD)

✅ 當出現 ≥1 組 Hard 候選，滿足:
- `IC ≥ 0.02`
- `IR ≥ 0.5`
- `PMR ≥ 0.55`
- `H ∈ {1, 2, 3}`
- `lag = 1h`
- 通過 No-Drift preflight

並且設為 `selected_source=base`，進入權重生成管道。

---

## Notes

- 所有 Soft / Shadow 實驗需標記 `_soft` / `_shadow` suffix
- 正式 KPI 只看 Hard + Base
- 每次實驗需記錄於 `RUNLOG_OPERATIONS.md`
- 所有 policy 變更需經過 No-Drift preflight 驗證
