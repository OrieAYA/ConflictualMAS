"""
scenario_builder.py
---------------------
自分でシナリオ（グリッドの大きさ・壁・ロボット・目標地点）を対話的に
作成するためのツール。プログラミングの知識がなくても、質問に答える
だけでオリジナルのシナリオを作れます。

使い方:
    python3 scenario_builder.py
    → 質問に順番に答えると、scenario_custom.json が作られます。

作ったシナリオを動画にする:
    python3 video_export.py --scenario scenario_custom.json 出力名.mp4

作ったシナリオをGUIで見る:
    python3 gui.py --scenario scenario_custom.json
"""

import json
import sys

from grid_world import GridWorld
from robots import Robot, Waypoint

DEFAULT_COLORS = ["#e6194B", "#4363d8", "#3cb44b", "#f58231", "#911eb4",
                   "#42d4f4", "#f032e6", "#bfef45"]


def ask(prompt, default=None, cast=str):
    suffix = f"（デフォルト: {default}）" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix} > ").strip()
        if raw == "" and default is not None:
            return default
        try:
            return cast(raw)
        except ValueError:
            print("  → 入力の形式が正しくありません。もう一度お願いします。")


def ask_cell(prompt, width, height, default=None):
    while True:
        raw = ask(prompt, default=(f"{default[0]},{default[1]}" if default else None))
        try:
            x_str, y_str = raw.split(",")
            x, y = int(x_str.strip()), int(y_str.strip())
        except ValueError:
            print("  → 「x,y」の形式で入力してください（例: 3,4）。")
            continue
        if not (0 <= x < width and 0 <= y < height):
            print(f"  → グリッドの範囲外です（0〜{width-1}, 0〜{height-1}）。")
            continue
        return (x, y)


def build_scenario_interactively():
    print("=" * 60)
    print("シナリオ作成ツール（質問に答えるだけでOKです）")
    print("=" * 60)

    width = ask("グリッドの横幅を入力してください", default=14, cast=int)
    height = ask("グリッドの縦幅を入力してください", default=10, cast=int)

    print("\n--- 障害物（壁）の設定 ---")
    print("「x,y」の形式でマスを1つずつ入力してください。")
    print("何も入力せず Enter だけ押すと、障害物の入力を終了します。")
    obstacles = set()
    while True:
        raw = input(f"障害物マス ({len(obstacles)}個入力済み) > ").strip()
        if raw == "":
            break
        try:
            x_str, y_str = raw.split(",")
            x, y = int(x_str.strip()), int(y_str.strip())
            if 0 <= x < width and 0 <= y < height:
                obstacles.add((x, y))
            else:
                print("  → グリッドの範囲外です。")
        except ValueError:
            print("  → 「x,y」の形式で入力してください（例: 5,3）。")

    print("\n--- ロボットの設定 ---")
    n_robots = ask("ロボットの台数を入力してください", default=4, cast=int)

    robots = []
    for i in range(n_robots):
        print(f"\n### ロボット {i + 1} / {n_robots} ###")
        name = ask("名前", default=f"ロボット{i + 1}")
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        print(f"  色は自動で割り当てます: {color}")
        start = ask_cell("スタート地点 (x,y)", width, height, default=(0, 0))

        n_wp = ask("このロボットの目標地点（ウェイポイント）の数", default=5, cast=int)
        waypoints = []
        for j in range(n_wp):
            cell = ask_cell(f"  目標地点 {j + 1} の座標 (x,y)", width, height)
            earliest = ask(f"  目標地点 {j + 1} が使用可能になる最短時刻 "
                            f"(0なら制約なし)", default=0, cast=int)
            label = ask(f"  目標地点 {j + 1} のラベル（表示名）", default=f"地点{j + 1}")
            waypoints.append(Waypoint(cell, earliest, label))

        robots.append(Robot(id=i, name=name, color=color, start=start, waypoints=waypoints))

    grid = GridWorld(width, height, obstacles)
    return grid, robots


def save_scenario_json(grid, robots, path):
    data = {
        "width": grid.width,
        "height": grid.height,
        "obstacles": sorted(list(grid.obstacles)),
        "robots": [
            {
                "id": r.id, "name": r.name, "color": r.color, "start": list(r.start),
                "waypoints": [
                    {"cell": list(wp.cell), "earliest_time": wp.earliest_time, "label": wp.label}
                    for wp in r.waypoints
                ],
            }
            for r in robots
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_scenario_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    obstacles = {tuple(c) for c in data["obstacles"]}
    grid = GridWorld(data["width"], data["height"], obstacles)
    robots = [
        Robot(
            id=r["id"], name=r["name"], color=r["color"], start=tuple(r["start"]),
            waypoints=[
                Waypoint(tuple(wp["cell"]), wp.get("earliest_time", 0), wp.get("label", ""))
                for wp in r["waypoints"]
            ],
        )
        for r in data["robots"]
    ]
    return grid, robots


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "scenario_custom.json"
    grid, robots = build_scenario_interactively()
    save_scenario_json(grid, robots, out_path)
    print(f"\nシナリオを保存しました: {out_path}")
    print("動画を作るには:")
    print(f"  python3 video_export.py --scenario {out_path} 出力名.mp4")
    print("GUIで見るには:")
    print(f"  python3 gui.py --scenario {out_path}")
