"""DMX 通道冲突核心扫描逻辑。

区间统一采用左闭右开 [start_ms, end_ms)：
- 仅端点相接（一个的 end_ms 等于另一个的 start_ms）不算冲突；
- 重叠段取两个起点中的较大值到两个终点中的较小值。
"""

from __future__ import annotations

from array import array
from bisect import bisect_right
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cue:
    cue: str
    channel: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Conflict:
    channel: int
    cue_a: str
    cue_b: str
    overlap_start_ms: int
    overlap_end_ms: int


@dataclass(frozen=True)
class ContentionWindow:
    """同一通道上一段连续抢值时段：由相交或首尾相接的冲突重叠区间合并而来。"""

    channel: int
    start_ms: int
    end_ms: int
    conflict_count: int


@dataclass(frozen=True)
class IsolationItem:
    """一条建议临时隔离的 cue：携带源数组下标、名称、通道与原始区间。"""

    index: int
    cue: str
    channel: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class ScanReport:
    channels_checked: int
    conflicts: list[Conflict] = field(default_factory=list)
    contention_windows: list[ContentionWindow] = field(default_factory=list)
    isolation_plan: list[IsolationItem] = field(default_factory=list)


def _time_order(cue: Cue) -> tuple[int, int, str]:
    """扫描顺序：必须按时间先后，扫描线提前终止才成立。"""
    return (cue.start_ms, cue.end_ms, cue.cue)


def _name_order(cue: Cue) -> tuple[str, int, int]:
    """同一冲突内两个 cue 的展示顺序：按名称字典序。"""
    return (cue.cue, cue.start_ms, cue.end_ms)


def scan_conflicts(cues: list[Cue]) -> ScanReport:
    """按通道找出全部两两重叠，返回确定排序的冲突列表。"""
    by_channel: dict[int, list[Cue]] = {}
    for cue in cues:
        by_channel.setdefault(cue.channel, []).append(cue)

    conflicts: list[Conflict] = []
    for channel, items in by_channel.items():
        # 必须按 start_ms 升序扫描：b.start_ms >= a.end_ms 时其后的 cue
        # 起点只会更晚，不可能再与 a 重叠，才能安全提前终止。
        # （若按名称排序，名称居中但时间很靠后的 cue 会触发提前终止，
        # 把名称靠后但实际重叠的 cue 跳过，导致漏判。）
        items.sort(key=_time_order)
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                if b.start_ms >= a.end_ms:
                    break
                overlap_start = max(a.start_ms, b.start_ms)
                overlap_end = min(a.end_ms, b.end_ms)
                if overlap_start >= overlap_end:
                    continue
                first, second = (a, b) if _name_order(a) <= _name_order(b) else (b, a)
                conflicts.append(
                    Conflict(
                        channel=channel,
                        cue_a=first.cue,
                        cue_b=second.cue,
                        overlap_start_ms=overlap_start,
                        overlap_end_ms=overlap_end,
                    )
                )

    conflicts.sort(
        key=lambda c: (
            c.channel,
            c.overlap_start_ms,
            c.overlap_end_ms,
            c.cue_a,
            c.cue_b,
        )
    )
    return ScanReport(
        channels_checked=len(by_channel),
        conflicts=conflicts,
        contention_windows=merge_contention_windows(conflicts),
        isolation_plan=plan_isolation(cues),
    )


def merge_contention_windows(conflicts: list[Conflict]) -> list[ContentionWindow]:
    """把同一通道内相交或首尾相接的冲突重叠区间确定性合并为抢值时间窗。

    区间仍按左闭右开处理：按 (起点, 终点) 排序后线性扫描，
    下一区间起点 <= 当前窗口终点（含端点相接）即并入，否则结算当前窗口。
    结果按通道 → 起点 → 终点排序（合并后同通道起点唯一，排序完全确定）。
    """
    by_channel: dict[int, list[Conflict]] = {}
    for conflict in conflicts:
        by_channel.setdefault(conflict.channel, []).append(conflict)

    windows: list[ContentionWindow] = []
    for channel, items in by_channel.items():
        items.sort(key=lambda c: (c.overlap_start_ms, c.overlap_end_ms))
        start: int | None = None
        end = 0
        count = 0
        for item in items:
            if start is not None and item.overlap_start_ms > end:
                windows.append(ContentionWindow(channel, start, end, count))
                start = None
            if start is None:
                start, end, count = item.overlap_start_ms, item.overlap_end_ms, 1
            else:
                end = max(end, item.overlap_end_ms)
                count += 1
        if start is not None:
            windows.append(ContentionWindow(channel, start, end, count))

    windows.sort(key=lambda w: (w.channel, w.start_ms, w.end_ms))
    return windows


def plan_isolation(cues: list[Cue]) -> list[IsolationItem]:
    """求隔离数量最少的临时隔离方案，使余下 cue 在各自通道内互不重叠。

    以每个通道的原始 cue 为候选全局求解（等价于加权区间调度）：
    - 隔离数最少 ⟺ 保留数最多；
    - 隔离总时长更短 ⟺ 保留总时长更长（候选总时长恒定）；
    - 仍并列时，被隔离下标的升序序列字典序更小者优，即优先隔离源数组下标小的 cue。

    逐冲突任选一端隔离的贪心在链式重叠（A∩B、B∩C 而 A∩C 为空）上会多隔离一条，
    因此这里按通道做动态规划取全局最优，再按通道 → 源下标排序输出。
    """
    by_channel: dict[int, list[tuple[int, Cue]]] = {}
    for index, cue in enumerate(cues):
        by_channel.setdefault(cue.channel, []).append((index, cue))

    plan: list[IsolationItem] = []
    for channel, items in by_channel.items():
        for index in _channel_isolation_indices(items):
            cue = cues[index]
            plan.append(
                IsolationItem(
                    index=index,
                    cue=cue.cue,
                    channel=channel,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                )
            )
    plan.sort(key=lambda item: (item.channel, item.index))
    return plan


def _channel_isolation_indices(items: list[tuple[int, Cue]]) -> list[int]:
    """单通道内的最优隔离下标（升序）。items 为 (源数组下标, cue) 对。"""
    m = len(items)
    if m < 2:
        return []

    # 按终点升序做加权区间调度；终点相同再按起点、源下标，保证扫描顺序确定。
    seq = sorted(items, key=lambda pair: (pair[1].end_ms, pair[1].start_ms, pair[0]))
    ends = [cue.end_ms for _, cue in seq]

    # 第三优先级“被隔离下标升序序列字典序更小”等价于
    # “保留集合（数量恒定为 m-k）的降序下标序列字典序更大”（取补集翻转）。
    # DP 只追踪保留集合：每次保留只是在兼容前缀的链尾追加一个区间，
    # 天然是单链追加而非位图并入，因此可用结构共享的持久化集合表示，
    # 工作集随 cue 数量线性增长，不再为每个 cue 保存 Θ(m) 位的大整数。
    ranks = _KeptSetRanks(seq)

    # dp[i]：只考虑 seq 前 i 个时的最优保留方案
    #   (保留数, 保留总时长, 保留集合节点)，逐元素 O(log m) 决胜。
    # decision[i]/parent[i] 记录最优选择以便结尾一次性回溯，
    # 二者总长度均为 O(m)，不保留任何二次方位图。
    count = [0] * (m + 1)
    kept_ms = [0] * (m + 1)
    node_of = [0] * (m + 1)
    decision = bytearray(m + 1)  # 1 = 保留 seq[i-1]；0 = 隔离
    parents = array("i", [-1]) * (m + 1)

    for i in range(1, m + 1):
        index, cue = seq[i - 1]
        duration = cue.end_ms - cue.start_ms
        # 半开区间：终点 <= 当前起点即兼容（端点相接可共存）。
        compatible = bisect_right(ends, cue.start_ms, 0, i - 1)
        # 方案一：隔离当前 cue，方案与 dp[i-1] 相同。
        skip = (count[i - 1], kept_ms[i - 1], node_of[i - 1])
        # 方案二：保留当前 cue：其按终点早于/等于它且相容的区间
        # （恰为 seq[:compatible] 中的最优保留集）之后追加当前项。
        keep_node = ranks.append(node_of[compatible], index)
        keep = (count[compatible] + 1, kept_ms[compatible] + duration, keep_node)
        if skip > keep:
            count[i], kept_ms[i], node_of[i] = skip
            decision[i] = 0
            parents[i] = i - 1
        else:
            count[i], kept_ms[i], node_of[i] = keep
            decision[i] = 1
            parents[i] = compatible

    # 沿决策链回溯：dp[i] 选保留则收集该 cue，再跳到其兼容前缀状态。
    dropped: list[int] = []
    i = m
    while i > 0:
        if decision[i]:
            i = parents[i]
        else:
            dropped.append(seq[i - 1][0])
            i -= 1
    dropped.sort()
    return dropped


class _KeptSet:
    """按源下标排序的持久化（结构共享）保留集合，仅支持“取某版本 + 追加一项”。

    叶子按源下标升序编号；节点严格对应树的固定层级（空子树统一为节点 0），
    内容相同的节点经 hash-consing 得到相同 id。于是第三层决胜
    “降序下标序列字典序更大”只需沿右子树优先下降到首个存在性不同的叶子，
    单次 O(log m)，无需还原集合或比较位图。

    m 次 append 恰好创建 m·log2(size) 个节点，每个节点在三个 array('i')
    中占 12 字节；规范化字典的键编码为单个整数。总工作集 O(m log m)
    （实际接近线性），不再出现总位数 Θ(m²) 的大整数位图。
    """

    __slots__ = ("size", "depth", "rank_of", "left", "right", "intern", "shift")

    def __init__(self, seq: list[tuple[int, Cue]]) -> None:
        m = len(seq)
        self.size = 1 << max(1, (m - 1).bit_length())
        self.depth = self.size.bit_length()
        # 源下标 → 叶子名次（下标越小名次越小）。
        self.rank_of = {
            index: rank
            for rank, (index, _) in enumerate(sorted(seq, key=lambda pair: pair[0]))
        }
        # 节点 0 表示任意层级的空子树（left/right 均指向自身）。
        self.left = array("i", [0])
        self.right = array("i", [0])
        # 每次 append 新建恰好 depth 个节点；据此给 intern 键留出足够位数。
        max_nodes = m * self.depth + 1
        self.shift = max(1, (max_nodes - 1).bit_length())
        self.intern: dict[int, int] = {}

    def _node(self, left: int, right: int) -> int:
        key = (left << self.shift) | right
        node = self.intern.get(key)
        if node is None:
            node = len(self.left)
            self.intern[key] = node
            self.left.append(left)
            self.right.append(right)
        return node

    def append(self, root: int, index: int) -> int:
        """返回在 root 版本中加入源下标 index 后的新版本根。"""
        target = self.rank_of[index]

        def build(node: int, lo: int, hi: int) -> int:
            if hi - lo == 1:
                return self._node(0, 0)  # 规范化叶子：右兄弟为 0 即表示叶子
            mid = (lo + hi) >> 1
            if target < mid:
                return self._node(build(self.left[node], lo, mid), self.right[node])
            return self._node(self.left[node], build(self.right[node], mid, hi))

        return build(root, 0, self.size)

    def prefer(self, winner: int, other: int) -> bool:
        """winner 的降序源下标序列是否字典序严格大于 other（同数量前提下决胜）。"""
        lo, hi = 0, self.size
        while winner != other and hi - lo > 1:
            mid = (lo + hi) >> 1
            wr, oth = self.right[winner], self.right[other]
            if wr != oth:
                # 右半侧（源下标更大的一半）已有差异，最高差异叶子必在其中。
                winner, other, lo = wr, oth, mid
            else:
                winner, other, hi = self.left[winner], self.left[other], mid
        return winner != 0
