"""
gui.py
-------
画面操作つきのデモ（Tkinter、追加インストール不要）。

配布・授業での上映には video_export.py で作ったMP4の方が手軽ですが、
その場でパラメータの効果を試したい場合はこちらを使ってください。

使い方:
    python3 gui.py                              # デフォルトシナリオ
    python3 gui.py --scenario scenario_custom.json   # 自作シナリオ
"""

import sys
import tkinter as tk
from tkinter import ttk

from scenario import build_default_scenario
from dbvns_planner import plan_all_robots

CELL = 46
MARGIN = 26

CAPTION_TEMPLATES = {
    "区間開始": "[出発] {name}：{目標} へ向かう",
    "前進": "[前進] → {cell} (t={t})",
    "到着": "[到着] {cell} に到着 (t={t})",
    "行き詰まり": "[行き詰まり] {cell}：周囲に進めるマスがない",
    "打ち切り": "[打ち切り] ステップ上限に到達、この枝を打ち切り",
    "禁止": "[Shake+Decompose] {起点} から {cell} を禁止",
    "有望branchなし": "[上限フィルタ] 条件を満たす枝が見つからない (k={k})",
    "改善": "[改善] より良い部分解が見つかった",
    "改善なし": "[改善なし] k={k}（次はもっと後退する）",
    "区間解決": "[区間解決] DbVNSがこの区間を解決 (t={t})",
    "目標未到達": "[限界] 反復回数を使い切り、目標に届かなかった",
    "経路確定": "[経路確定] {name} の経路が確定（{長さ}ステップ）",
}


class MapfDemoApp(tk.Tk):
    def __init__(self, scenario_path=None):
        super().__init__()
        self.title("多ロボット経路計画デモ（DbVNSに時空間制約を統合）")
        self.configure(bg="#1e1e2e")

        if scenario_path:
            from scenario_builder import load_scenario_json
            self.grid_world, self.robots = load_scenario_json(scenario_path)
        else:
            self.grid_world, self.robots = build_default_scenario()

        self.paths, self.log = plan_all_robots(self.grid_world, self.robots)
        self.robots_by_id = {r.id: r for r in self.robots}

        self.event_index = 0
        self.active_robot_id = None
        self.explorer_item = None
        self.exploration_dots = {r.id: [] for r in self.robots}
        self.forbidden_marks = {r.id: [] for r in self.robots}
        self.confirmed_lines = {}

        self.mode = "planning"
        self.autoplay = False
        self.sim_running = False
        self.sim_t = 0
        self.sim_horizon = max((len(p) - 1 for p in self.paths.values()), default=0)

        self._build_layout()
        self._draw_static_grid()
        self._draw_waypoints()
        self._draw_robot_tokens()
        self._log_line("ようこそ。「次のステップ」を押すと、各ロボットがDbVNSで"
                        "経路をどう探索するか、1手ずつ確認できます。")

    # ------------------------------------------------------------------
    def _build_layout(self):
        main = tk.Frame(self, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=10, pady=10)

        canvas_w = MARGIN * 2 + self.grid_world.width * CELL
        canvas_h = MARGIN * 2 + self.grid_world.height * CELL
        self.canvas = tk.Canvas(main, width=canvas_w, height=canvas_h,
                                 bg="#282a3a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=3, padx=(0, 12))

        right = tk.Frame(main, bg="#1e1e2e")
        right.grid(row=0, column=1, sticky="n")

        tk.Label(right, text="多ロボット経路計画デモ", font=("Helvetica", 14, "bold"),
                 bg="#1e1e2e", fg="white").pack(anchor="w")

        self.phase_label = tk.Label(right, text="フェーズ：経路計画中",
                                     font=("Helvetica", 11, "italic"),
                                     bg="#1e1e2e", fg="#8be9fd")
        self.phase_label.pack(anchor="w", pady=(2, 10))

        btns1 = tk.Frame(right, bg="#1e1e2e")
        btns1.pack(anchor="w", pady=(0, 6))
        tk.Button(btns1, text="次のステップ ▶", width=18,
                  command=self.step_planning).grid(row=0, column=0, padx=2, pady=2)
        self.btn_auto = tk.Button(btns1, text="自動再生 ⏵", width=18,
                                   command=self.toggle_autoplay)
        self.btn_auto.grid(row=0, column=1, padx=2, pady=2)
        tk.Button(btns1, text="計画を最後まで進める ⏭", width=18,
                  command=self.skip_planning).grid(row=1, column=0, padx=2, pady=2)
        tk.Button(btns1, text="リセット ⟲", width=18,
                  command=self.reset_all).grid(row=1, column=1, padx=2, pady=2)

        speed_frame = tk.Frame(right, bg="#1e1e2e")
        speed_frame.pack(anchor="w", pady=(4, 10), fill="x")
        tk.Label(speed_frame, text="速さ：", bg="#1e1e2e", fg="white").pack(side="left")
        self.speed_scale = tk.Scale(speed_frame, from_=400, to=10, orient="horizontal",
                                     length=200, showvalue=False, bg="#1e1e2e", fg="white",
                                     troughcolor="#44475a", highlightthickness=0)
        self.speed_scale.set(120)
        self.speed_scale.pack(side="left", padx=6)

        tk.Frame(right, bg="#44475a", height=2).pack(fill="x", pady=8)

        btns2 = tk.Frame(right, bg="#1e1e2e")
        btns2.pack(anchor="w", pady=(0, 10))
        tk.Button(btns2, text="シミュレーション開始", width=18,
                  command=self.start_simulation, bg="#50fa7b").grid(row=0, column=0, padx=2, pady=2)
        tk.Button(btns2, text="一時停止", width=18,
                  command=self.pause_simulation).grid(row=0, column=1, padx=2, pady=2)

        legend = tk.LabelFrame(right, text="ロボットと目標地点", bg="#1e1e2e", fg="white")
        legend.pack(anchor="w", fill="x", pady=(4, 10))
        for r in self.robots:
            line = tk.Frame(legend, bg="#1e1e2e")
            line.pack(anchor="w", fill="x")
            swatch = tk.Canvas(line, width=14, height=14, bg="#1e1e2e", highlightthickness=0)
            swatch.pack(side="left", padx=(2, 6))
            swatch.create_oval(1, 1, 13, 13, fill=r.color, outline="")
            wp_text = "、".join(
                f"{wp.label}" + (f"(t≥{wp.earliest_time})" if wp.earliest_time else "")
                for wp in r.waypoints
            )
            tk.Label(line, text=f"{r.name} → {wp_text}", bg="#1e1e2e", fg="white",
                     font=("Helvetica", 8)).pack(side="left")

        log_frame = tk.LabelFrame(right, text="探索の記録", bg="#1e1e2e", fg="white")
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, width=48, height=16, bg="#282a3a", fg="#f8f8f2",
                                 wrap="word", font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.status_label = tk.Label(right, text="", bg="#1e1e2e", fg="#f1fa8c",
                                      font=("Helvetica", 10, "bold"))
        self.status_label.pack(anchor="w", pady=(8, 0))

    # ------------------------------------------------------------------
    def _cell_center(self, cell):
        x, y = cell
        return (MARGIN + x * CELL + CELL / 2, MARGIN + y * CELL + CELL / 2)

    def _cell_bbox(self, cell):
        x, y = cell
        x0, y0 = MARGIN + x * CELL, MARGIN + y * CELL
        return (x0, y0, x0 + CELL, y0 + CELL)

    def _draw_static_grid(self):
        for x in range(self.grid_world.width):
            for y in range(self.grid_world.height):
                x0, y0, x1, y1 = self._cell_bbox((x, y))
                fill = "#44475a" if (x, y) in self.grid_world.obstacles else "#282a3a"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#3b3d4d")

    def _draw_waypoints(self):
        for r in self.robots:
            for i, wp in enumerate(r.waypoints, start=1):
                cx, cy = self._cell_center(wp.cell)
                self.canvas.create_rectangle(cx - 10, cy - 10, cx + 10, cy + 10,
                                              outline=r.color, width=2, dash=(3, 2))
                self.canvas.create_text(cx, cy, text=str(i), fill=r.color,
                                         font=("Helvetica", 9, "bold"))
                if wp.earliest_time:
                    self.canvas.create_text(cx, cy + 17, text=f"t≥{wp.earliest_time}",
                                             fill=r.color, font=("Helvetica", 6))

    def _draw_robot_tokens(self):
        self.robot_tokens = {}
        for r in self.robots:
            cx, cy = self._cell_center(r.start)
            oval = self.canvas.create_oval(cx - 13, cy - 13, cx + 13, cy + 13,
                                            fill=r.color, outline="white", width=2)
            label = self.canvas.create_text(cx, cy, text=r.name[-1], fill="white",
                                             font=("Helvetica", 10, "bold"))
            self.robot_tokens[r.id] = (oval, label)

    def _move_robot_token(self, robot_id, cell):
        cx, cy = self._cell_center(cell)
        oval, label = self.robot_tokens[robot_id]
        self.canvas.coords(oval, cx - 13, cy - 13, cx + 13, cy + 13)
        self.canvas.coords(label, cx, cy)

    # ------------------------------------------------------------------
    def _log_line(self, text):
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

    def _format_event(self, event):
        robot = self.robots_by_id[event["robot_id"]]
        template = CAPTION_TEMPLATES.get(event["kind"], event["kind"])
        data = dict(event)
        data["name"] = robot.name
        try:
            return f"[{robot.name}] " + template.format(**data)
        except KeyError:
            return f"[{robot.name}] {event['kind']} {event}"

    # ------------------------------------------------------------------
    def step_planning(self):
        if self.mode != "planning":
            return
        if self.event_index >= len(self.log.events):
            self.status_label.config(text="全ロボットの計画が完了しました。")
            self.autoplay = False
            return

        event = self.log.events[self.event_index]
        self.event_index += 1
        self._apply_planning_event(event)
        self._log_line(self._format_event(event))

        if self.event_index >= len(self.log.events):
            self.status_label.config(text="全ロボットの計画が完了しました。"
                                           "シミュレーションを開始できます。")
            self.autoplay = False

    def _apply_planning_event(self, event):
        rid = event["robot_id"]
        robot = self.robots_by_id[rid]

        if rid != self.active_robot_id:
            self.active_robot_id = rid
            self.phase_label.config(text=f"フェーズ：経路計画 — {robot.name}")

        kind = event["kind"]

        if kind == "区間開始":
            self._clear_exploration(rid)

        if "cell" in event and kind in ("前進", "区間開始", "到着", "行き詰まり", "打ち切り"):
            cx, cy = self._cell_center(event["cell"])
            if self.explorer_item is not None:
                self.canvas.delete(self.explorer_item)
            color = "#ff5555" if kind in ("行き詰まり", "打ち切り") else robot.color
            self.explorer_item = self.canvas.create_oval(
                cx - 8, cy - 8, cx + 8, cy + 8, outline=color, width=3
            )
            dot = self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                                           fill=robot.color, outline="")
            self.exploration_dots[rid].append(dot)

        if kind == "禁止" and "cell" in event:
            cx, cy = self._cell_center(event["cell"])
            l1 = self.canvas.create_line(cx - 7, cy - 7, cx + 7, cy + 7, fill="#ff5555", width=2)
            l2 = self.canvas.create_line(cx - 7, cy + 7, cx + 7, cy - 7, fill="#ff5555", width=2)
            self.forbidden_marks[rid].extend([l1, l2])

        if kind == "経路確定":
            self._clear_exploration(rid)
            self._draw_confirmed_path(rid)

    def _clear_exploration(self, rid):
        if self.explorer_item is not None:
            self.canvas.delete(self.explorer_item)
            self.explorer_item = None
        for item in self.exploration_dots[rid]:
            self.canvas.delete(item)
        self.exploration_dots[rid] = []
        for item in self.forbidden_marks[rid]:
            self.canvas.delete(item)
        self.forbidden_marks[rid] = []

    def _draw_confirmed_path(self, rid):
        robot = self.robots_by_id[rid]
        path = self.paths[rid]
        points = []
        for cell, _t in path:
            points.extend(self._cell_center(cell))
        if len(points) >= 4:
            line = self.canvas.create_line(*points, fill=robot.color, width=3, arrow="last")
            self.confirmed_lines[rid] = line
        self._move_robot_token(rid, robot.start)

    def skip_planning(self):
        self.autoplay = False
        while self.event_index < len(self.log.events):
            self.step_planning()

    def toggle_autoplay(self):
        self.autoplay = not self.autoplay
        self.btn_auto.config(text="一時停止 ⏸" if self.autoplay else "自動再生 ⏵")
        if self.autoplay:
            self._autoplay_tick()

    def _autoplay_tick(self):
        if not self.autoplay or self.mode != "planning":
            return
        self.step_planning()
        if self.event_index < len(self.log.events):
            self.after(int(self.speed_scale.get()), self._autoplay_tick)
        else:
            self.autoplay = False
            self.btn_auto.config(text="自動再生 ⏵")

    # ------------------------------------------------------------------
    def start_simulation(self):
        self.skip_planning()
        self.mode = "simulation"
        self.phase_label.config(text="フェーズ：シミュレーション（実際に移動）")
        self.sim_t = 0
        self.sim_running = True
        self._log_line("——— シミュレーション開始 ———")
        self._simulation_tick()

    def pause_simulation(self):
        self.sim_running = False

    def _position_at(self, rid, t):
        path = self.paths[rid]
        idx = min(t, len(path) - 1)
        return path[idx][0]

    def _simulation_tick(self):
        if not self.sim_running or self.mode != "simulation":
            return
        for r in self.robots:
            self._move_robot_token(r.id, self._position_at(r.id, self.sim_t))
        self.status_label.config(text=f"シミュレーション中 — t = {self.sim_t} / {self.sim_horizon}")
        if self.sim_t >= self.sim_horizon:
            self.sim_running = False
            self.status_label.config(text="シミュレーション終了 — 全ロボットが到着しました。")
            return
        self.sim_t += 1
        self.after(int(self.speed_scale.get()) * 2, self._simulation_tick)

    # ------------------------------------------------------------------
    def reset_all(self):
        self.canvas.delete("all")
        self.autoplay = False
        self.sim_running = False
        self.mode = "planning"
        self.event_index = 0
        self.active_robot_id = None
        self.explorer_item = None
        self.exploration_dots = {r.id: [] for r in self.robots}
        self.forbidden_marks = {r.id: [] for r in self.robots}
        self.confirmed_lines = {}
        self.log_text.delete("1.0", "end")
        self.phase_label.config(text="フェーズ：経路計画中")
        self.status_label.config(text="")
        self._draw_static_grid()
        self._draw_waypoints()
        self._draw_robot_tokens()
        self._log_line("リセットしました。「次のステップ」からやり直せます。")


def _parse_scenario_arg(argv):
    if "--scenario" in argv:
        idx = argv.index("--scenario")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return None


if __name__ == "__main__":
    app = MapfDemoApp(scenario_path=_parse_scenario_arg(sys.argv))
    app.mainloop()
