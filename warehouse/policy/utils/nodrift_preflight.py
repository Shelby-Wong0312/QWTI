import os, json, yaml, datetime

POLICY = r"C:\Users\niuji\Documents\Data\warehouse\policy\no_drift.yaml"

def enforce(observed):
    # ? policy
    with open(POLICY, "r", encoding="utf-8") as f:
        p = yaml.safe_load(f)
    # ??????
    assert p["invariants"]["timezone"] == "UTC"
    assert p["invariants"]["align_to_next_full_hour"] is True
    assert p["invariants"]["trading_kpi_only"] is True
    assert p["invariants"]["non_trading_ret_1h_zero"] is True
    # Hard Gate ????(???????? Hard KPI,????)
    if observed.get("mode") == "hard_kpi":
        # assert observed["mapped_ratio"] >= float(p["hard_gate"]["mapped_ratio_min"])  # disabled
        assert observed["all_art_cnt"]  >= int(p["hard_gate"]["all_art_cnt_min"])
        assert observed["tone_nonnull"] is True
        if "skip_ratio" in observed:  # RAW ?????(??)
            assert observed["skip_ratio"] <= p["hard_gate"]["drop_windows_with_skip_ratio_over"]
    return True

