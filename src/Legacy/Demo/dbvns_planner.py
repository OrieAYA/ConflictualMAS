"""
dbvns_planner.py
------------------
論文 (Crepin Simon Franck & Shohei Yokoyama,
"A Distributed Multi-Meta Agent MTTDS Solver") の
Decomposition-based VNS（DbVNS、Algorithm 1）を、できるだけ
"改変せずに" そのまま使うモジュールです。

【重要な設計方針】
これはMAPF専用アルゴリズムではありません。論文のDbVNSの構造
（貪欲構築 → Shake → Decompose → 上限フィルタ → 受理判定）は
一切変えていません。変えているのはただ1点だけです：

    近傍候補を数え上げる箇所（論文の memory.check_neighborhood に
    相当する _candidate_neighbors 関数）に、
    「この時刻にこの場所へ移動してよいか？」
    という制約を1つ追加しているだけです。

この制約は、DbVNSがもともと持っている「forbidden（禁止集合）」の
仕組みに自然に組み込まれます。つまり、他のロボットに予約されている
時空間マスは、すでに探索済みで失敗したマスと同じように「候補から
除外」されるだけで、アルゴリズム自体の反復・後退・分解のロジックは
論文のものと同一です。この制約を追加するだけで、探索結果はより
現実的（＝他のロボットと衝突しない）になり、その意味でより
"最適" な解に近づきます。

【論文とこの実装の対応】
  - "POI" → グリッド上のマス (x, y) （時刻 t 付き）
  - "報酬（最大化）" → 目的地までの距離（最小化）＝報酬の符号を反転
    したものと考える
  - "禁止されたPOI" → すでに試して失敗したマス、または
    他ロボットに予約されているマス
  - Agent::greedy_construction → greedy_construction()
  - Agent::decompose_with_forbidden → decompose_with_forbidden()
  - Shake（k に応じて後退する量） → 論文の式をそのまま踏襲
  - 上限フィルタ（δ による足切り） → 同じ考え方を距離の文脈に変換

パラメータは、論文セクション6.2.2の実験設定に忠実にしています：
  k_max = 4、反復回数 = 100、連続空振り上限 = 3、
  δ（max_divergence） = 0.3、max_decompositions = 5
"""

from dataclasses import dataclass, field
from reservation import ReservationTable


# ============================================================
# 論文セクション6.2.2の実験パラメータ
# ============================================================
K_MAX = 4
MAX_ITERATIONS = 100
MAX_CONSECUTIVE_EMPTY = 3
DELTA_MAX_DIVERGENCE = 0.3  # δ
MAX_DECOMPOSITIONS = 5


@dataclass
class SearchNode:
    """論文の DecomposedSolution に相当：探索木の1ノード（部分経路）。"""
    cell: tuple
    t: int
    parent: "SearchNode" = None
    forbidden: frozenset = field(default_factory=frozenset)

    def depth(self):
        d, node = 0, self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    def path(self):
        """根までさかのぼって完全な経路（(cell, t) のリスト）を復元する。"""
        out = []
        node = self
        while node is not None:
            out.append((node.cell, node.t))
            node = node.parent
        out.reverse()
        return out


class PlanningLog:
    """可視化（GUI・動画）用に、探索の一部始終を記録する日誌。"""

    def __init__(self):
        self.events = []

    def add(self, robot_id, kind, **data):
        self.events.append({"robot_id": robot_id, "kind": kind, **data})


def _candidate_neighbors(grid, node, goal, reservations, robot_id):
    """
    論文の memory.check_neighborhood に相当。近傍マスのうち、
      (1) このノードで既に禁止されているもの
      (2) 【追加した制約】この時刻にそのマスへ移動できないもの
          （他ロボットに予約されている＝衝突する）
    を除外し、目的地に近い順に並べ替えて返す。
    """
    candidates = []
    for nc in grid.neighbors(node.cell):
        if nc in node.forbidden:
            continue
        if not reservations.is_move_valid(node.cell, nc, node.t, robot_id):
            continue  # ここが「この時刻にこの場所へ移動できるか」の判定
        candidates.append(nc)
    candidates.sort(key=lambda c: (grid.true_distance(c, goal), c == node.cell))
    return candidates


def greedy_construction(node, grid, goal, earliest_time, reservations, robot_id, log,
                         max_extra_steps=80):
    """論文の Agent::greedy_construction。目的地に着くか、
    行き詰まる（候補が尽きる）まで貪欲にマスを進める。"""
    current = node
    steps = 0
    while True:
        if current.cell == goal and current.t >= earliest_time:
            log.add(robot_id, "到着", cell=current.cell, t=current.t)
            return current

        if steps >= max_extra_steps:
            log.add(robot_id, "打ち切り", cell=current.cell, t=current.t)
            return current

        candidates = _candidate_neighbors(grid, current, goal, reservations, robot_id)
        if not candidates:
            log.add(robot_id, "行き詰まり", cell=current.cell, t=current.t)
            return current  # 行き詰まりノード → Decomposeが必要

        next_cell = candidates[0]
        current = SearchNode(next_cell, current.t + 1, current)
        log.add(robot_id, "前進", cell=next_cell, t=current.t)
        steps += 1


def decompose_with_forbidden(stuck_node, log, robot_id):
    """論文の Agent::decompose_with_forbidden。行き詰まったノードから
    根に向かってさかのぼり、各祖先ノードで「そこから進んで失敗した
    マス」を禁止集合に追加する。祖先ノードそれぞれが新しい候補branch
    になる。"""
    branches = []
    current = stuck_node
    while current.parent is not None:
        failed_cell = current.cell
        current = current.parent
        current.forbidden = current.forbidden | {failed_cell}
        branches.append(current)
        log.add(robot_id, "禁止", cell=failed_cell, 起点=current.cell, t=current.t)
    return branches


def plan_segment(grid, start_cell, start_t, goal_cell, earliest_time, reservations,
                  robot_id, log,
                  k_max=K_MAX, max_iterations=MAX_ITERATIONS,
                  max_consecutive_empty=MAX_CONSECUTIVE_EMPTY,
                  delta=DELTA_MAX_DIVERGENCE, max_decompositions=MAX_DECOMPOSITIONS):
    """
    1台のロボットの1区間（現在地 → 次のウェイポイント）を計画する。
    論文 Algorithm 1（DbVNS）のメインループをそのまま再現している。
    """
    root = SearchNode(start_cell, start_t, None, frozenset())
    log.add(robot_id, "区間開始", cell=start_cell, t=start_t, 目標=goal_cell)

    current = greedy_construction(root, grid, goal_cell, earliest_time, reservations,
                                   robot_id, log)
    if current.cell == goal_cell and current.t >= earliest_time:
        return current.path()

    # ------------------------------------------------------------
    # 貪欲構築だけでは行き詰まった → Algorithm 1 のメインループ
    # (Shake / Decompose / 上限フィルタ / 貪欲構築 / 受理判定)
    # ------------------------------------------------------------
    pbest = current
    pbest_score = grid.true_distance(pbest.cell, goal_cell)  # 小さいほど良い
    k = 1
    iteration = 0
    consecutive_empty = 0

    while iteration < max_iterations:
        # --- A. SHAKE：論文の式をそのまま踏襲 ---
        # until = min( (size // k_max) * (k-1), |2 - size| )
        size = pbest.depth() + 1
        until = min((size // k_max) * (k - 1), abs(2 - size))
        shaken = pbest
        for _ in range(until):
            if shaken.parent is not None:
                shaken = shaken.parent

        # --- B. DECOMPOSE：候補branchを生成 ---
        branches = decompose_with_forbidden(shaken, log, robot_id)

        if not branches:
            k += 1
            iteration += 1
            consecutive_empty += 1
            if consecutive_empty >= max_consecutive_empty:
                break
            continue

        # --- C. 上限フィルタ（δ に基づく足切り）---
        # 論文: local_threshold = pbest_fitness * (1 - δ) （報酬は大きいほど良い）
        # ここでは距離なので逆に、「pbest_score / (1-δ) 以下」を「有望」とする
        threshold = pbest_score / (1 - delta)
        promising = [b for b in branches if grid.true_distance(b.cell, goal_cell) <= threshold]

        if not promising:
            k += 1
            iteration += 1
            consecutive_empty += 1
            log.add(robot_id, "有望branchなし", k=k)
            if consecutive_empty >= max_consecutive_empty:
                break
            continue

        if len(promising) > max_decompositions:
            promising.sort(key=lambda b: grid.true_distance(b.cell, goal_cell))
            promising = promising[:max_decompositions]

        consecutive_empty = 0

        # --- D. 各branchを貪欲構築で伸ばす ---
        success_node = None
        best_partial = None
        best_partial_score = pbest_score

        for branch in promising:
            rebuilt = greedy_construction(branch, grid, goal_cell, earliest_time,
                                           reservations, robot_id, log)
            if rebuilt.cell == goal_cell and rebuilt.t >= earliest_time:
                success_node = rebuilt
                break
            score = grid.true_distance(rebuilt.cell, goal_cell)
            if score < best_partial_score:
                best_partial_score = score
                best_partial = rebuilt

        if success_node is not None:
            log.add(robot_id, "区間解決", t=success_node.t)
            return success_node.path()

        # --- E. 受理判定 ---
        if best_partial is not None and best_partial_score < pbest_score:
            pbest = best_partial
            pbest_score = best_partial_score
            k = 1
            log.add(robot_id, "改善", cell=pbest.cell, score=pbest_score)
        else:
            k = min(k + 1, k_max)
            log.add(robot_id, "改善なし", k=k)

        iteration += 1

    # ------------------------------------------------------------
    # 反復回数の予算を使い切った場合：論文自身が示す通り（6.3.1節）
    # DbVNSは常に最適解を見つけるわけではない。ここでは正直に、
    # それまでに見つかった最善の部分経路をそのまま返す（無理に
    # 別のアルゴリズムで解こうとはしない）。
    # ------------------------------------------------------------
    log.add(robot_id, "目標未到達", cell=pbest.cell, t=pbest.t)
    return pbest.path()


def plan_all_robots(grid, robots, horizon=250, **dbvns_kwargs):
    """
    すべてのロボットを、リストの順番（＝優先度順）で計画する
    （Priority Planning）。先に計画されたロボットの経路は
    ReservationTable に確定として登録され、後から計画される
    ロボットは、_candidate_neighbors の中でその制約を自動的に
    考慮することになる。

    戻り値:
      paths : dict  robot_id -> (cell, t) のリスト（完全な経路）
      log   : PlanningLog（GUI・動画で1ステップずつ再生するための記録）
    """
    reservations = ReservationTable()
    log = PlanningLog()
    paths = {}

    for robot in robots:
        full_path = [(robot.start, 0)]
        cursor_cell, cursor_t = robot.start, 0

        for wp in robot.waypoints:
            segment = plan_segment(
                grid, cursor_cell, cursor_t, wp.cell, wp.earliest_time,
                reservations, robot.id, log, **dbvns_kwargs,
            )
            full_path.extend(segment[1:])  # 接続点の重複を避ける
            cursor_cell, cursor_t = full_path[-1]

        paths[robot.id] = full_path
        reservations.reserve_path(full_path, robot.id, horizon=horizon)
        log.add(robot.id, "経路確定", 長さ=len(full_path))

    return paths, log
