#!/usr/bin/env python
"""
簡化版 BaseWeightAllocator

用途：
- 給 hourly_monitor.py import 用，避免 ModuleNotFoundError
- 提供一個穩定的「信號  權重」轉換邏輯
- 支援多種常見方法名稱：
    - allocate
    - get_weight
    - compute_weight
    - calculate_position
    - __call__
"""

from dataclasses import dataclass
from typing import Optional

import math
import pandas as pd


@dataclass
class BaseWeightAllocatorConfig:
    """
    基本配置：
    - max_abs_weight：單一策略最大絕對權重（例如 0.15 = 15%）
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
        if "max_weight" in kwargs:
            config.max_abs_weight = float(kwargs["max_weight"])
        if "base_weight" in kwargs:
            # base_weight 目前不影響計算，僅記錄
            self.base_weight = float(kwargs["base_weight"])
        else:
            self.base_weight = 0.05
        if "scale" in kwargs:
            config.scale = float(kwargs["scale"])

        self.config = config

    # ------------------------
    # 核心邏輯
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

        # 縮放後丟進 tanh：[-, +]  [-1, +1]
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

    def __call__(self, signal: Optional[float]) -> float:
        """允許 allocator( signal ) 這種呼叫方式。"""
        return self.allocate(signal)

    def log_position(self, timestamp: str, prediction: float, position: float,
                     features: Optional[dict] = None, metadata: Optional[dict] = None) -> dict:
        """
        記錄倉位資訊，供 hourly_monitor.py 使用。
        回傳一個 dict 供寫入 CSV。
        """
        record = {
            'timestamp': timestamp,
            'prediction': prediction,
            'position': position,
            'max_weight': self.config.max_abs_weight,
        }
        if metadata:
            record.update(metadata)
        return record
