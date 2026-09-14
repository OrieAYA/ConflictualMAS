"""
test_headless.py
------------------
画面表示なしでアルゴリズムを検証するスクリプト。
新しいシナリオを作った時に「衝突がないか」「全ロボットが目標に
到達できたか」を素早く確認するために使う。

実行: python3 test_headless.py
"""

from scenario import build_default_scenario
from dbvns_planner import plan_all_robots


def check_no_collisions(paths):
    vertex = {}
    edge = {}
    problems = []
    for rid, path in paths.items():
        for i, (cell, t) in enumerate(path):
            key = (cell, t)
            if key in vertex and vertex[key] != rid:
                problems.append(f"頂点衝突: ロボット{rid}と{vertex[key]}が{cell}でt={t}に衝突")
            vertex[key] = rid
            if i > 0:
                prev_cell, prev_t = path[i - 1]
                ekey = (prev_cell, cell, prev_t)
                rev = (cell, prev_cell, prev_t)
                if rev in edge and edge[rev] != rid:
                    problems.append(
                        f"すれ違い衝突: ロボット{rid}と{edge[rev]}が{prev_cell}⇔{cell}でt={prev_t}"
                    )
                edge[ekey] = rid
    return problems


if __name__ == "__main__":
    grid, robots = build_default_scenario()
    paths, log = plan_all_robots(grid, robots)

    for rid, path in paths.items():
        robot = next(r for r in robots if r.id == rid)
        goal_reached = path[-1][0] == robot.waypoints[-1].cell
        print(f"--- {robot.name} (id={rid}) : {len(path)} ステップ, "
              f"最終目標到達={'OK' if goal_reached else 'NG'} ---")

    problems = check_no_collisions(paths)
    print("\n=== 衝突チェック ===")
    if problems:
        for p in problems:
            print("問題:", p)
    else:
        print("衝突は検出されませんでした。OK。")

    print(f"\n日誌の総イベント数: {len(log.events)}")
