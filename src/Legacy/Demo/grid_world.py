"""
grid_world.py
-------------
環境（グリッド）を表すモジュール。

論文 (Crepin & Yokoyama, "A Distributed Multi-Meta Agent MTTDS Solver")
のセクション2にある環境定義 E = (A, V, P)（A: ノード、V: 重み付き辺、
P: 興味地点の集合）を、格子（グリッド）に単純化したものです。

`true_distance` メソッドは、論文の Global Memory（セクション5.1、式8）
にある「最近傍POI探索・経路キャッシュ」の役割に相当します。論文では
Dijkstra ベースの探索で近傍POIを事前計算していますが、ここでは目的地
ごとに幅優先探索（BFS）で「壁を迂回した本当の最短距離」を1回だけ計算
してキャッシュしておき、DbVNSの貪欲構築がその値を目安（ヒューリスティ
ック）として使います。
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class GridWorld:
    width: int
    height: int
    obstacles: set = field(default_factory=set)  # 障害物マスの集合 (x, y)
    _distance_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def in_bounds(self, cell):
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, cell):
        return self.in_bounds(cell) and cell not in self.obstacles

    def neighbors(self, cell):
        """上下左右の4方向 + 「その場で待つ」（1タイムステップ静止）。"""
        x, y = cell
        candidates = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in candidates if self.is_free(c)]

    def true_distance(self, cell, goal):
        """
        壁を迂回した「本当の最短距離」（BFSで一度だけ計算しキャッシュ）。
        論文のGlobal Memory（memory.check_path / check_neighborhood）が
        果たす役割と同じで、DbVNSの貪欲構築が「次にどのマスへ進むべきか」
        を判断する際の目安として使います。
        """
        if goal not in self._distance_cache:
            self._distance_cache[goal] = self._bfs_from(goal)
        dist_map = self._distance_cache[goal]
        return dist_map.get(cell, self.width * self.height)  # 到達不能なら大きな値

    def _bfs_from(self, goal):
        dist = {goal: 0}
        queue = deque([goal])
        while queue:
            cur = queue.popleft()
            x, y = cur
            for nb in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if self.is_free(nb) and nb not in dist:
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        return dist
