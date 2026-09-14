"""
video_export.py
-----------------
このデモを1本のMP4動画として書き出すスクリプト。
Tkinterのような画面操作なしで、そのまま生徒に見せられる形式です。

構成：
  1. イントロ（タイトルカード）
  2. 計画フェーズ：ロボットごとに、DbVNSが経路を探索していく様子を
     1イベントずつ再生（探索中のマス・禁止されたマス・完成した経路）
  3. 「計画完了」の切り替えカード（全ロボットの経路を一望できる）
  4. シミュレーションフェーズ：全ロボットが実際に同時に動く様子を、
     滑らかな補間つきでアニメーション。時刻を大きく表示し、計画済みの
     経路は薄く表示したまま残すので、「今どこを進んでいるか」が
     はっきり分かる。
  5. アウトロ

使い方:
    python3 video_export.py [出力ファイル名.mp4]
"""

import sys
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

from scenario import build_default_scenario
from dbvns_planner import plan_all_robots

FPS = 12
TITLE_HOLD = int(FPS * 1.2)
RECAP_HOLD = int(FPS * 2.0)
OUTRO_HOLD = int(FPS * 2.5)
SIM_SUBFRAMES = 5
SIM_HOLD_END = int(FPS * 2.0)
TRAIL_LEN = 4

CAPTION_TEMPLATES = {
    "区間開始": "[出発] {name}：{目標} へ向かう",
    "前進": "[前進] → {cell} (t={t})",
    "到着": "[到着] {cell} に到着 (t={t})",
    "行き詰まり": "[行き詰まり] {cell}：周囲に進めるマスがない",
    "打ち切り": "[打ち切り] ステップ上限に到達、この枝を打ち切り",
    "禁止": "[Shake + Decompose] {起点} から {cell} を禁止",
    "有望branchなし": "[上限フィルタ] 条件を満たす枝が見つからない (k={k})",
    "改善": "[改善] より良い部分解が見つかった",
    "改善なし": "[改善なし] k={k}（次はもっと後退する）",
    "区間解決": "[区間解決] DbVNSがこの区間を解決 (t={t})",
    "目標未到達": "[限界] 反復回数を使い切り、目標に届かなかった（DbVNSの限界）",
    "経路確定": "[経路確定] {name} の経路が確定（{長さ}ステップ）",
}


def format_caption(event, robots_by_id):
    robot = robots_by_id[event["robot_id"]]
    template = CAPTION_TEMPLATES.get(event["kind"], event["kind"])
    data = dict(event)
    data["name"] = robot.name
    try:
        return f"[{robot.name}] " + template.format(**data)
    except KeyError:
        return f"[{robot.name}] {event['kind']}"


def hold(spec, n):
    return [spec] * n


# ======================================================================
# フレーム列の構築（計画フェーズ）
# ======================================================================
def build_planning_frames(robots, robots_by_id, paths, log):
    frames = []
    confirmed = {}
    active_rid = None
    trail = []
    forbidden = []

    for event in log.events:
        rid = event["robot_id"]
        kind = event["kind"]
        robot = robots_by_id[rid]

        if rid != active_rid:
            active_rid = rid
            trail, forbidden = [], []
            frames.extend(hold({
                "type": "title", "text": f"{robot.name} の計画を開始します",
                "sub": "DbVNS：貪欲構築 → Shake → Decompose → 上限フィルタ",
            }, TITLE_HOLD))

        if kind == "区間開始":
            trail, forbidden = [], []

        if "cell" in event and kind in ("前進", "区間開始", "到着", "行き詰まり", "打ち切り"):
            if kind != "区間開始":
                trail.append(event["cell"])
            frames.append({
                "type": "step", "robot": rid,
                "explorer": event["cell"],
                "blocked": kind in ("行き詰まり", "打ち切り"),
                "trail": list(trail), "forbidden": list(forbidden),
                "confirmed": dict(confirmed),
                "caption": format_caption(event, robots_by_id),
            })
        elif kind == "禁止":
            forbidden.append(event["cell"])
            frames.append({
                "type": "step", "robot": rid, "explorer": None, "blocked": False,
                "trail": list(trail), "forbidden": list(forbidden),
                "confirmed": dict(confirmed),
                "caption": format_caption(event, robots_by_id),
            })
        else:
            frames.append({
                "type": "step", "robot": rid, "explorer": None, "blocked": False,
                "trail": list(trail), "forbidden": list(forbidden),
                "confirmed": dict(confirmed),
                "caption": format_caption(event, robots_by_id),
            })

        if kind == "経路確定":
            confirmed[rid] = paths[rid]
            trail, forbidden = [], []

    return frames


# ======================================================================
# フレーム列の構築（シミュレーションフェーズ）
# ======================================================================
def build_simulation_frames(robots, paths):
    horizon = max(len(p) - 1 for p in paths.values())

    frames = hold({
        "type": "recap", "confirmed": paths,
        "caption": "計画完了：全ロボットの経路が確定しました。これから同時に動かします。",
    }, RECAP_HOLD)

    for t in range(horizon):
        for s in range(SIM_SUBFRAMES):
            alpha = s / SIM_SUBFRAMES
            positions, trails = {}, {}
            for r in robots:
                path = paths[r.id]
                idx0 = min(t, len(path) - 1)
                idx1 = min(t + 1, len(path) - 1)
                c0, c1 = path[idx0][0], path[idx1][0]
                positions[r.id] = (c0[0] + (c1[0] - c0[0]) * alpha,
                                    c0[1] + (c1[1] - c0[1]) * alpha)
                trails[r.id] = [path[i][0] for i in range(max(0, t - TRAIL_LEN), idx0 + 1)]
            frames.append({
                "type": "sim", "positions": positions, "trails": trails,
                "confirmed": paths, "t": t, "horizon": horizon,
            })

    final_positions = {r.id: paths[r.id][-1][0] for r in robots}
    final_trails = {r.id: [c for c, _t in paths[r.id][-TRAIL_LEN:]] for r in robots}
    frames.extend(hold({
        "type": "sim", "positions": final_positions, "trails": final_trails,
        "confirmed": paths, "t": horizon, "horizon": horizon,
    }, SIM_HOLD_END))
    return frames, horizon


# ======================================================================
# 描画
# ======================================================================
class Renderer:
    def __init__(self, grid, robots):
        self.grid = grid
        self.robots = robots
        self.robots_by_id = {r.id: r for r in robots}

        self.fig = plt.figure(figsize=(15, 8.4), facecolor="#1e1e2e")
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.03)
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_panel = self.fig.add_subplot(gs[0, 1])

        self._setup_static()
        self.dynamic_artists = []

    def _setup_static(self):
        w, h = self.grid.width, self.grid.height
        ax = self.ax
        ax.set_facecolor("#282a3a")
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks(range(w + 1))
        ax.set_yticks(range(h + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color="#3b3d4d", linewidth=0.6)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for (x, y) in self.grid.obstacles:
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#44475a", edgecolor="#3b3d4d"))

        for r in self.robots:
            for i, wp in enumerate(r.waypoints, start=1):
                cx, cy = wp.cell[0] + 0.5, wp.cell[1] + 0.5
                ax.add_patch(Rectangle((wp.cell[0] + 0.12, wp.cell[1] + 0.12), 0.76, 0.76,
                                        fill=False, edgecolor=r.color, linewidth=1.6,
                                        linestyle=(0, (3, 2))))
                ax.text(cx, cy - 0.05, str(i), color=r.color, ha="center", va="center",
                        fontsize=9, fontweight="bold")
                if wp.earliest_time:
                    ax.text(cx, cy + 0.32, f"t≥{wp.earliest_time}", color=r.color,
                            ha="center", va="center", fontsize=6)
            sx, sy = r.start[0] + 0.5, r.start[1] + 0.5
            ax.add_patch(Circle((sx, sy), 0.16, facecolor="none", edgecolor=r.color,
                                 linewidth=1.6))

        self.ax_panel.axis("off")
        self.ax_panel.set_xlim(0, 1)
        self.ax_panel.set_ylim(0, 1)
        self.ax_panel.set_facecolor("#1e1e2e")
        self.fig.patch.set_facecolor("#1e1e2e")

        self.ax_panel.text(0.02, 0.97, "多ロボット経路計画デモ", color="white",
                            fontsize=15, fontweight="bold", va="top")
        self.ax_panel.text(0.02, 0.925, "DbVNS（Crepin & Yokoyama論文）を応用",
                            color="#8be9fd", fontsize=9.5, va="top", style="italic")

        y0 = 0.86
        for r in self.robots:
            self.ax_panel.add_patch(Circle((0.045, y0), 0.014, transform=self.ax_panel.transAxes,
                                            facecolor=r.color, edgecolor="none", clip_on=False))
            labels = "、".join(wp.label for wp in r.waypoints)
            self.ax_panel.text(0.08, y0, r.name, color="white", fontsize=9.5,
                                fontweight="bold", va="center")
            self.ax_panel.text(0.08, y0 - 0.028, labels, color="#bdbdd0", fontsize=6.6,
                                va="center", wrap=True)
            y0 -= 0.075

        self.phase_text = self.ax_panel.text(0.02, y0 - 0.02, "", color="#f1fa8c",
                                              fontsize=10.5, fontweight="bold", va="top")
        self.caption_text = self.ax_panel.text(
            0.02, y0 - 0.09, "", color="#f8f8f2", fontsize=9.2, va="top", wrap=True
        )
        self.time_text = self.ax_panel.text(0.02, 0.05, "", color="#50fa7b", fontsize=13,
                                             fontweight="bold", va="bottom")

        self.title_text = ax.text(w / 2, h / 2, "", color="white", fontsize=17,
                                   fontweight="bold", ha="center", va="center")
        self.subtitle_text = ax.text(w / 2, h / 2 + 0.7, "", color="#8be9fd", fontsize=10,
                                      ha="center", va="center")

    def _clear_dynamic(self):
        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists = []
        self.title_text.set_text("")
        self.subtitle_text.set_text("")

    def render(self, frame):
        self._clear_dynamic()
        kind = frame["type"]

        if kind == "title":
            self.title_text.set_text(frame["text"])
            self.subtitle_text.set_text(frame.get("sub", ""))
            self.phase_text.set_text("フェーズ：計画")
            self.caption_text.set_text("")
            self.time_text.set_text("")
            return

        if kind == "intro":
            self.title_text.set_text(frame["text"])
            self.subtitle_text.set_text(frame.get("sub", ""))
            self.phase_text.set_text("")
            self.caption_text.set_text("")
            self.time_text.set_text("")
            return

        if kind == "outro":
            self.title_text.set_text(frame["text"])
            self.subtitle_text.set_text(frame.get("sub", ""))
            self.phase_text.set_text("フェーズ：完了")
            self.caption_text.set_text("")
            self.time_text.set_text("")
            self._draw_confirmed(frame.get("confirmed", {}))
            return

        # 計画済み経路（薄い線）は常に背景として描く
        self._draw_confirmed(frame.get("confirmed", {}))

        if kind == "step":
            robot = self.robots_by_id[frame["robot"]]
            self.phase_text.set_text(f"フェーズ：計画 —  {robot.name}")
            self.caption_text.set_text(frame["caption"])
            self.time_text.set_text("")

            for cell in frame["trail"]:
                cx, cy = cell[0] + 0.5, cell[1] + 0.5
                dot = Circle((cx, cy), 0.07, facecolor=robot.color, edgecolor="none", alpha=0.55)
                self.ax.add_patch(dot)
                self.dynamic_artists.append(dot)

            for cell in frame["forbidden"]:
                cx, cy = cell[0] + 0.5, cell[1] + 0.5
                l1, = self.ax.plot([cx - 0.18, cx + 0.18], [cy - 0.18, cy + 0.18],
                                    color="#ff5555", linewidth=2)
                l2, = self.ax.plot([cx - 0.18, cx + 0.18], [cy + 0.18, cy - 0.18],
                                    color="#ff5555", linewidth=2)
                self.dynamic_artists.extend([l1, l2])

            if frame["explorer"] is not None:
                cx, cy = frame["explorer"][0] + 0.5, frame["explorer"][1] + 0.5
                color = "#ff5555" if frame["blocked"] else robot.color
                ring = Circle((cx, cy), 0.28, facecolor="none", edgecolor=color, linewidth=2.6)
                self.ax.add_patch(ring)
                self.dynamic_artists.append(ring)

        elif kind == "recap":
            self.phase_text.set_text("フェーズ：計画完了 → シミュレーションへ")
            self.caption_text.set_text(frame["caption"])
            self.time_text.set_text("")

        elif kind == "sim":
            self.phase_text.set_text("フェーズ：シミュレーション（実際に同時に移動）")
            self.caption_text.set_text("経路は薄い線で表示中。丸が実際のロボットです。")
            self.time_text.set_text(f"t = {frame['t']} / {frame['horizon']}")

            for r in self.robots:
                for cell in frame["trails"].get(r.id, []):
                    cx, cy = cell[0] + 0.5, cell[1] + 0.5
                    dot = Circle((cx, cy), 0.10, facecolor=r.color, edgecolor="none", alpha=0.30)
                    self.ax.add_patch(dot)
                    self.dynamic_artists.append(dot)

            for r in self.robots:
                x, y = frame["positions"][r.id]
                cx, cy = x + 0.5, y + 0.5
                token = Circle((cx, cy), 0.34, facecolor=r.color, edgecolor="white", linewidth=1.8)
                self.ax.add_patch(token)
                self.dynamic_artists.append(token)
                label = self.ax.text(cx, cy, r.name[-1], color="white", fontsize=10,
                                      fontweight="bold", ha="center", va="center")
                self.dynamic_artists.append(label)

    def _draw_confirmed(self, confirmed):
        for rid, path in confirmed.items():
            robot = self.robots_by_id[rid]
            xs = [c[0] + 0.5 for c, _t in path]
            ys = [c[1] + 0.5 for c, _t in path]
            line, = self.ax.plot(xs, ys, color=robot.color, linewidth=2.0, alpha=0.55, zorder=1)
            self.dynamic_artists.append(line)


def build_all_frames(grid, robots, paths, log):
    robots_by_id = {r.id: r for r in robots}
    frames = []
    frames.extend(hold({
        "type": "intro",
        "text": "多ロボット経路計画デモ",
        "sub": "DbVNS（分解ベースVNS）に時空間制約を統合",
    }, int(FPS * 2.0)))

    frames.extend(build_planning_frames(robots, robots_by_id, paths, log))

    sim_frames, horizon = build_simulation_frames(robots, paths)
    frames.extend(sim_frames)

    all_reached = all(paths[r.id][-1][0] == r.waypoints[-1].cell for r in robots)
    outro_text = "全ロボットが衝突せずに到着しました" if all_reached else \
        "シミュレーション終了（一部のロボットは目標未到達 — DbVNSの限界の例）"
    frames.extend(hold({
        "type": "outro", "text": outro_text, "sub": "",
        "confirmed": paths,
    }, OUTRO_HOLD))

    return frames


def _parse_args(argv):
    scenario_path = None
    out_path = "mapf_demo.mp4"
    rest = []
    i = 1
    while i < len(argv):
        if argv[i] == "--scenario" and i + 1 < len(argv):
            scenario_path = argv[i + 1]
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    if rest:
        out_path = rest[0]
    return scenario_path, out_path


def main():
    scenario_path, out_path = _parse_args(sys.argv)

    if scenario_path:
        from scenario_builder import load_scenario_json
        grid, robots = load_scenario_json(scenario_path)
    else:
        grid, robots = build_default_scenario()

    paths, log = plan_all_robots(grid, robots)

    frames = build_all_frames(grid, robots, paths, log)
    print(f"総フレーム数: {len(frames)} (約 {len(frames)/FPS:.1f} 秒, {FPS}fps)")

    renderer = Renderer(grid, robots)

    def update(i):
        renderer.render(frames[i])
        return []

    anim = animation.FuncAnimation(renderer.fig, update, frames=len(frames),
                                    interval=1000 / FPS, blit=False)

    writer = animation.FFMpegWriter(fps=FPS, bitrate=2400)
    anim.save(out_path, writer=writer, dpi=110)
    print(f"書き出し完了: {out_path}")


if __name__ == "__main__":
    main()
