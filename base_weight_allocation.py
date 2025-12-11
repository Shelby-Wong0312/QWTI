cd "C:\Users\niuji\Documents\Data"

# 確保 monitoring 資料夾存在
if (-not (Test-Path "warehouse\monitoring")) {
    New-Item -ItemType Directory -Path "warehouse\monitoring" -Force | Out-Null
}

@'
#!/usr/bin/env python
"""
簡化版 BaseWeightAllocator

用途：
- 給 hourly_monitor.py import 用，避免 ModuleNotFoundError
- 提供一個穩定的「信號 ➜ 權重」轉換邏輯
- 支援多種常見方法名稱：
    - allocate
    - get_weight
    - compute_weight
    - calculate_position
    - log_position
    - __call__
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict

import math
import pandas as pd


@dataclass
class BaseWeightAllocatorConfig:
    """
    基本配置：
    - max_abs_weight：單一策略最大絕對權重（例如 0.15 = ±15%）
    - scale: 把原始 signal 放大/縮小用，避免太敏感
    """

    max_abs_weight: float = 0.15
    scale: float = 1.0


class BaseWeightAllocator:
    """
    極簡權重分配器：

    - 輸入一個連續信號（例如 預測報酬 / score）
    - 先做縮放，再用 tanh 壓在 [-1, 1]
    - 最後乘上 max_abs_weight，變成 [-max_abs_weight, +max_abs_weight]

    這樣可以避免信號太大直接給滿倉。
    """

    def __init__(self, config: Optional[BaseWeightAllocatorConfig] = None, **kwargs):
        if config is None:
            config = BaseWeightAllocatorConfig()

        # 允許用 kwargs 覆蓋
        if "max_abs_weight" in kwargs:
            config.max_abs_weight = float(kwargs["max_abs_weight"])
        if "scale" in kwargs:
            config.scale = float(kwargs["scale"])

        self.config = config

    # ------------------------
    # 核心邏輯：signal ➜ weight
    # ------------------------
    def allocate(self, signal: Optional[float]) -> float:
        """
        主要對外函式：輸入 signal，輸出建議權重（-max ~ +max）
        """
        if signal is None:
            return 0.0

        try:
            x = float(signal)
        except (TypeError, ValueError):
            return 0.0

        if pd.isna(x):
            return 0.0

        # 縮放後丟進 tanh：[-∞, +∞] → [-1, +1]
        scaled = x * self.config.scale
        tanh_val = math.tanh(scaled)

        weight = tanh_val * self.config.max_abs_weight

        # 再做一次 safety clip，避開意外
        max_w = abs(self.config.max_abs_weight)
        if weight > max_w:
            weight = max_w
        elif weight < -max_w:
            weight = -max_w

        return weight

    # ------------------------
    # 兼容不同名稱的呼叫方式
    # ------------------------
    def get_weight(self, signal: Optional[float]) -> float:
        """有些程式可能叫這個名稱。"""
        return self.allocate(signal)

    def compute_weight(self, signal: Optional[float]) -> float:
        """也可能叫 compute_weight。"""
        return self.allocate(signal)

    def calculate_position(self, signal: Optional[float]) -> float:
        """
        給 hourly_monitor.py 用的名稱：
        position = allocator.calculate_position(prediction)
        """
        return self.allocate(signal)

    def log_position(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        給 hourly_monitor.py 用來產生「寫進 positions.csv 的紀錄」。
        實作策略：盡量無腦相容，不假設欄位長什麼樣：

        - 如果呼叫時有 kwargs：原樣回傳 kwargs（最常見的情況）
        - 如果只有 positional arguments：
            - 我們盡量猜一些欄位名稱，否則就用 col_0, col_1... 收起來
        """
        if kwargs:
            # 最保險作法：直接把關鍵字參數當作一筆 record 回傳
            return dict(kwargs)

        record: Dict[str, Any] = {}

        # 沒有 kwargs，盡量幫忙猜欄位
        if len(args) >= 1:
            record["timestamp"] = args[0]
        if len(args) >= 2:
            record["position"] = args[1]
        if len(args) >= 3:
            record["signal"] = args[2]

        # 多出來的 args，用 col_2, col_3... 收起來
        for idx, extra in enumerate(args[3:], start=3):
            record[f"col_{idx}"] = extra

        return record

    def __call__(self, signal: Optional[float]) -> float:
        """允許 allocator( signal ) 這種呼叫方式。"""
        return self.allocate(signal)
'@ | Set-Content -Encoding UTF8 "warehouse\monitoring\base_weight_allocation.py"

# 確認本機檔案存在
Get-Item "warehouse\monitoring\base_weight_allocation.py"
