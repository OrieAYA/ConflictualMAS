"""
main.py
--------
エントリーポイント。

    python3 main.py                                   # デフォルトシナリオ
    python3 main.py --scenario scenario_custom.json    # 自作シナリオ

Tkinter が必要です（Windows/Macの公式インストーラには同梱。
Linux Debian/Ubuntuでは `sudo apt install python3-tk` が必要な場合あり）。
"""

import sys
from gui import MapfDemoApp, _parse_scenario_arg

if __name__ == "__main__":
    app = MapfDemoApp(scenario_path=_parse_scenario_arg(sys.argv))
    app.mainloop()
