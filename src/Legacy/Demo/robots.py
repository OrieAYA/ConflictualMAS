"""
robots.py
----------
ロボット（エージェント）と、その目標地点（ウェイポイント）の定義。

各ロボットは、順番に（在庫順に）複数のウェイポイントを訪れます。
各ウェイポイントには、必要なら「earliest_time」（その時刻より前には
到達が認められない）を設定できます。例えば「荷物がt=6にならないと
準備できない」といった、時間に依存する目標を表現するためです。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Waypoint:
    cell: tuple
    earliest_time: int = 0
    label: str = ""  # 例："荷物A"。表示用のラベル


@dataclass
class Robot:
    id: int
    name: str
    color: str
    start: tuple
    waypoints: List[Waypoint] = field(default_factory=list)
