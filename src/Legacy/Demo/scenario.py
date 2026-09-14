"""
scenario.py
------------
デモ用シナリオ：14×10の倉庫フロアに、中央の障害物ブロックといくつかの
柱があり、4台のロボットがそれぞれ5つの目標地点（ウェイポイント）を
順番に巡回します。目標地点はフロア全体に分散して配置されており、
経路が互いに交差する場面が自然に生まれるようにしています。

このファイルだけを編集すれば、授業用に自由にシナリオを作り直せます。
自分で対話的にシナリオを作りたい場合は scenario_builder.py を使って
ください。
"""

from grid_world import GridWorld
from robots import Robot, Waypoint


def build_default_scenario():
    width, height = 14, 10

    # 中央の障害物ブロック（建物の柱・部屋のようなもの）+ 散在する障害物
    obstacles = set()
    for x in range(5, 9):
        for y in range(3, 7):
            obstacles.add((x, y))
    for extra in [(2, 6), (11, 3), (9, 8), (4, 1), (12, 6)]:
        obstacles.add(extra)

    grid = GridWorld(width, height, obstacles)

    # フロア全体に分散させた「興味地点（POI）」プール
    P_NW = (1, 1)
    P_NE = (12, 1)
    P_SW = (1, 8)
    P_SE = (12, 8)
    P_N = (6, 0)
    P_S = (7, 9)
    P_W = (0, 4)
    P_E = (13, 5)
    P_C1 = (3, 4)   # 中央ブロックの西側
    P_C2 = (10, 4)  # 中央ブロックの東側
    P_C3 = (6, 8)   # 中央ブロックの南側
    P_C4 = (7, 1)   # 中央ブロックの北側

    robots = [
        Robot(
            id=0, name="ロボット赤", color="#e6194B", start=(0, 0),
            waypoints=[
                Waypoint(P_NE, 0, "地点1"),
                Waypoint(P_C2, 4, "地点2"),
                Waypoint(P_S, 0, "地点3"),
                Waypoint(P_W, 0, "地点4"),
                Waypoint(P_C4, 0, "地点5"),
            ],
        ),
        Robot(
            id=1, name="ロボット青", color="#4363d8", start=(13, 9),
            waypoints=[
                Waypoint(P_NW, 0, "地点1"),
                Waypoint(P_C1, 0, "地点2"),
                Waypoint(P_N, 6, "地点3"),
                Waypoint(P_E, 0, "地点4"),
                Waypoint(P_C3, 0, "地点5"),
            ],
        ),
        Robot(
            id=2, name="ロボット緑", color="#3cb44b", start=(0, 9),
            waypoints=[
                Waypoint(P_SE, 0, "地点1"),
                Waypoint(P_C4, 0, "地点2"),
                Waypoint(P_W, 5, "地点3"),
                Waypoint(P_C2, 0, "地点4"),
                Waypoint(P_N, 0, "地点5"),
            ],
        ),
        Robot(
            id=3, name="ロボット橙", color="#f58231", start=(13, 0),
            waypoints=[
                Waypoint(P_SW, 0, "地点1"),
                Waypoint(P_C3, 0, "地点2"),
                Waypoint(P_E, 0, "地点3"),
                Waypoint(P_C1, 7, "地点4"),
                Waypoint(P_S, 0, "地点5"),
            ],
        ),
    ]

    return grid, robots
