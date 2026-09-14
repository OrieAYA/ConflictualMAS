"""
reservation.py
---------------
「時空間の予約テーブル」。

これがまさに、今回追加したい制約そのものです：
    ある場所 (x, y) に、ある時刻 t に、移動してよいかどうか？

DbVNSの本体（dbvns_planner.py）は、論文のオリジナルの構造
（貪欲構築 → Shake → Decompose → 上限フィルタ）をそのまま使い、
近傍候補を数え上げる箇所（論文の memory.check_neighborhood に相当）
で、このテーブルに問い合わせて「予約されていないマスだけ」を候補に
残します。つまりMAPF専用の別アルゴリズムを組んでいるのではなく、
DbVNSの「forbidden（禁止集合）」の仕組みに、時空間の制約を1つ
追加しているだけです。

具体的に防ぐのは2種類の衝突：
- 頂点衝突：2台のロボットが同じ時刻に同じマスにいる
- すれ違い衝突：2台のロボットが同じ時刻に位置を入れ替える
  （例：ロボットAが(2,2)→(3,2)、ロボットBが(3,2)→(2,2)を同時に行う）
"""

from dataclasses import dataclass, field


@dataclass
class ReservationTable:
    # (x, y, t) -> robot_id
    vertex: dict = field(default_factory=dict)
    # (x1, y1, x2, y2, t) -> robot_id  （時刻tに(x1,y1)から(x2,y2)へ移動）
    edge: dict = field(default_factory=dict)

    def is_vertex_free(self, cell, t, robot_id):
        occ = self.vertex.get((cell[0], cell[1], t))
        return occ is None or occ == robot_id

    def is_edge_free(self, from_cell, to_cell, t, robot_id):
        # すれ違い禁止：逆方向の移動が同時刻に予約されていないか確認
        occ = self.edge.get((to_cell[0], to_cell[1], from_cell[0], from_cell[1], t))
        return occ is None or occ == robot_id

    def is_move_valid(self, from_cell, to_cell, t, robot_id):
        """t = 出発時刻。to_cell には t+1 に到着する。
        これが「この時刻にこの場所へ移動できるか」の判定そのもの。"""
        return (
            self.is_vertex_free(to_cell, t + 1, robot_id)
            and self.is_edge_free(from_cell, to_cell, t, robot_id)
        )

    def reserve_path(self, path, robot_id, horizon=None):
        """
        path : そのロボットが確定した経路（(cell, t) のリスト）。
        各マス・各辺を予約し、さらに最後のマスに horizon まで
        「駐車」させておく（後から計画される優先度の低いロボットが
        その場所を通り抜けてしまわないようにするため）。
        """
        for i, (cell, t) in enumerate(path):
            self.vertex[(cell[0], cell[1], t)] = robot_id
            if i > 0:
                prev_cell, prev_t = path[i - 1]
                self.edge[(prev_cell[0], prev_cell[1], cell[0], cell[1], prev_t)] = robot_id

        if horizon is not None and path:
            last_cell, last_t = path[-1]
            for t in range(last_t + 1, horizon + 1):
                self.vertex[(last_cell[0], last_cell[1], t)] = robot_id
