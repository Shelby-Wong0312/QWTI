import os, json, yaml, datetime

POLICY = r"C:\Users\niuji\Documents\Data\warehouse\policy\no_drift.yaml"

def enforce(observed):
    # 讀 policy
    with open(POLICY, "r", encoding="utf-8") as f:
        p = yaml.safe_load(f)
    # 不可變硬規範
    assert p["invariants"]["timezone"] == "UTC"
    assert p["invariants"]["align_to_next_full_hour"] is True
    assert p["invariants"]["trading_kpi_only"] is True
    assert p["invariants"]["non_trading_ret_1h_zero"] is True
    # Hard Gate 最低條件（若你這支腳本會產 Hard KPI，就要檢查）
    if observed.get("mode") == "hard_kpi":
        assert observed["mapped_ratio"] >= float(p["hard_gate"]["mapped_ratio_min"])
        assert observed["all_art_cnt"]  >= int(p["hard_gate"]["all_art_cnt_min"])
        assert observed["tone_nonnull"] is True
        if "skip_ratio" in observed:  # RAW 解析健康度（可選）
            assert observed["skip_ratio"] <= p["hard_gate"]["drop_windows_with_skip_ratio_over"]
    return True
