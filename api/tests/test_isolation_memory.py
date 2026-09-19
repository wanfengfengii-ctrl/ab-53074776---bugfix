"""内存回归：完全无冲突的扫描不得为空的 isolation_plan 建立二次方决胜结构。

修复前 `_channel_isolation_indices` 为同通道每个 cue 保存一个位权任意精度整数
（总位数 Θ(m²)），并逐 DP 前缀保留位图状态；输入从 8,000 翻倍到 16,000 条时，
`scan_conflicts` 期间的 Python 分配峰值增长约 3.29 倍。修复后决胜状态改为
线性空间的可回溯链，峰值应随 cue 数量近似线性增长（阈值取 3.0 留有余量）。
"""

import gc
import tracemalloc

from app.scanner import Cue, scan_conflicts


def _peak_during_scan(n: int) -> int:
    """扫描 n 条同通道、互不重叠的合法 cue，返回 scan_conflicts 期间的分配峰值。"""
    cues = [Cue(str(i), 1, i * 2, i * 2 + 1) for i in range(n)]
    tracemalloc.start()
    report = scan_conflicts(cues)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # 输入彼此不重叠：不得报告冲突，也不得隔离任何 cue
    assert report.conflicts == []
    assert report.isolation_plan == []
    return peak


def test_conflict_free_scan_peak_memory_scales_near_linearly():
    small = _peak_during_scan(8000)
    gc.collect()
    large = _peak_during_scan(16000)
    ratio = large / small
    assert ratio <= 3.0, (small, large, ratio)
