#!/usr/bin/env python3
"""
fastlio2_gui.py — FASTLIO2 Control Panel
Dark-themed PyQt5 GUI for launching mapping / localization stacks
and calling ROS2 services without a terminal.

Run:
    python3 fastlio2_gui.py
Or after colcon build + source:
    ros2 run fastlio2_bringup fastlio2_gui.py
"""

import os
import sys
import html
import signal
import subprocess
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog,
    QCheckBox, QPlainTextEdit, QFrame, QGroupBox,
    QSizePolicy, QScrollBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QColor, QTextCursor, QFont, QLinearGradient

# ── Catppuccin Mocha palette ──────────────────────────────────────────────────
C = {
    "base":     "#1e1e2e",
    "mantle":   "#181825",
    "crust":    "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "overlay0": "#6c7086",
    "overlay1": "#7f849c",
    "text":     "#cdd6f4",
    "subtext1": "#bac2de",
    "subtext0": "#a6adc8",
    "blue":     "#89b4fa",
    "sapphire": "#74c7ec",
    "sky":      "#89dceb",
    "teal":     "#94e2d5",
    "green":    "#a6e3a1",
    "yellow":   "#f9e2af",
    "peach":    "#fab387",
    "red":      "#f38ba8",
    "mauve":    "#cba6f7",
    "lavender": "#b4befe",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {C["base"]};
    color: {C["text"]};
    font-family: 'Ubuntu', 'Noto Sans', 'Segoe UI', sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {C["mantle"]};
    border: 1px solid {C["surface1"]};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    color: {C["overlay1"]};
    font-weight: bold;
    font-size: 11px;
    letter-spacing: 1.5px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: {C["mantle"]};
}}
QPushButton {{
    background-color: {C["surface0"]};
    color: {C["text"]};
    border: 1px solid {C["surface1"]};
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {C["surface1"]};
    border-color: {C["overlay0"]};
}}
QPushButton:pressed {{
    background-color: {C["surface2"]};
}}
QPushButton:disabled {{
    background-color: {C["surface0"]};
    color: {C["overlay0"]};
    border-color: {C["surface0"]};
}}
QLineEdit {{
    background-color: {C["crust"]};
    color: {C["text"]};
    border: 1px solid {C["surface1"]};
    border-radius: 5px;
    padding: 5px 10px;
    font-size: 13px;
}}
QLineEdit:focus {{
    border-color: {C["blue"]};
    background-color: {C["mantle"]};
}}
QCheckBox {{
    color: {C["subtext1"]};
    spacing: 8px;
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {C["surface2"]};
    background: {C["crust"]};
}}
QCheckBox::indicator:checked {{
    background: {C["blue"]};
    border-color: {C["blue"]};
    image: none;
}}
QCheckBox::indicator:disabled {{
    background: {C["surface0"]};
    border-color: {C["surface1"]};
}}
QPlainTextEdit {{
    background-color: {C["crust"]};
    color: {C["subtext1"]};
    border: 1px solid {C["surface1"]};
    border-radius: 6px;
    padding: 6px 8px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.4;
}}
QScrollBar:vertical {{
    background: {C["mantle"]};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {C["surface2"]};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {C["overlay0"]};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: {C["mantle"]};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {C["surface2"]};
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
QFrame#hline {{
    background-color: {C["surface0"]};
    max-height: 1px;
    border: none;
}}
QFrame#accent_line {{
    background-color: {C["blue"]};
    max-height: 2px;
    border: none;
}}
"""

# ── Service definitions ───────────────────────────────────────────────────────

def _parse_reloc_response(raw):
    """Extract pose + fitness from a GlobalRelocalize service response."""
    import re
    fitness = re.search(r'fitness_score=([\d.eE+\-]+)', raw)
    x       = re.search(r'\bx=([-\d.eE+\-]+)',         raw)
    y       = re.search(r'\by=([-\d.eE+\-]+)',         raw)
    yaw     = re.search(r'yaw_deg=([-\d.eE+\-]+)',     raw)
    if fitness and x and y and yaw:
        return (f"Relocalized  |  fitness: {float(fitness.group(1)):.3f}  "
                f"|  x: {float(x.group(1)):.2f} m  y: {float(y.group(1)):.2f} m  "
                f"yaw: {float(yaw.group(1)):.1f}°")
    return raw


_CHECK_LOC = {
    "id":      "check_status",
    "label":   "Check Localization",
    "icon":    "✅",
    "color":   C["sapphire"],
    "srv":     "/localizer/is_valid",
    "iface":   "interface/srv/IsValid",
    "args":    lambda path: "{}",
    "timeout": 10,
}

SERVICES = {
    "mapping": [
        {
            "id":      "save_map",
            "label":   "Save Map",
            "icon":    "💾",
            "color":   C["teal"],
            "srv":     "/fastlio2/save_map",
            "iface":   "interface/srv/SaveMaps",
            "args":    lambda path: f"{{file_path: '{path}', save_patches: false}}",
            "timeout": 120,
        },
        _CHECK_LOC,
    ],
    "localization": [
        {
            "id":      "start_mapping",
            "label":   "Start Map Update",
            "icon":    "🗺",
            "color":   C["green"],
            "srv":     "/localizer/start_mapping",
            "iface":   "interface/srv/StartMapping",
            "args":    lambda path: "{output_path: '', voxel_size: 0.05}",
            "timeout": 10,
        },
        {
            "id":      "stop_mapping",
            "label":   "Stop & Merge Map",
            "icon":    "💾",
            "color":   C["yellow"],
            "srv":     "/localizer/stop_mapping",
            "iface":   "interface/srv/StopMapping",
            "args":    lambda path: "{output_path: ''}",
            "timeout": 30,
        },
        {
            "id":             "relocalize",
            "label":          "Force Re-Localize",
            "icon":           "🎯",
            "color":          C["mauve"],
            "srv":            "/localizer/global_relocalize",
            "iface":          "interface/srv/GlobalRelocalize",
            "args":           lambda path: "{pcd_path: '', force: true}",
            "timeout":        120,   # global search over large maps can take >20 s
            "parse_response": _parse_reloc_response,
        },
        _CHECK_LOC,
    ],
}


# ── Background threads ────────────────────────────────────────────────────────

class ServiceThread(QThread):
    """Calls a ros2 service in a background thread and emits the result."""
    result = pyqtSignal(str, bool, str)  # label, ok, display_message

    def __init__(self, label, cmd, srv_name, timeout=20, parse_response=None):
        super().__init__()
        self._label          = label
        self._cmd            = cmd
        self._srv_name       = srv_name
        self._timeout        = timeout
        self._parse_response = parse_response

    def run(self):
        try:
            proc = subprocess.run(
                self._cmd, capture_output=True, text=True, timeout=self._timeout,
                env={**os.environ},
            )
            raw = (proc.stdout + proc.stderr).strip()
            ok  = proc.returncode == 0 and "success=false" not in raw.lower()
            # Use a custom parser to produce a readable one-liner when available
            display = self._parse_response(raw) if (ok and self._parse_response) else raw
            self.result.emit(self._label, ok, display)
        except subprocess.TimeoutExpired:
            self.result.emit(
                self._label, False,
                f"Timed out waiting for service ({self._timeout} s)."
            )
        except Exception as exc:
            self.result.emit(self._label, False, str(exc))


class LogReader(QThread):
    """Forwards process stdout lines to the GUI."""
    line = pyqtSignal(str)

    def __init__(self, proc):
        super().__init__()
        self._proc = proc

    def run(self):
        try:
            for raw in self._proc.stdout:
                self.line.emit(raw.rstrip())
        except Exception:
            pass
        self.line.emit("─── process output ended ───")


# ── Status dot widget ─────────────────────────────────────────────────────────

class StatusDot(QWidget):
    def __init__(self, size=12, parent=None):
        super().__init__(parent)
        self._color = QColor(C["surface2"])
        self._size  = size
        self.setFixedSize(size + 2, size + 2)

    def set_color(self, hex_color):
        self._color = QColor(hex_color)
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # Subtle glow
        glow = QColor(self._color)
        glow.setAlpha(50)
        p.setBrush(glow)
        p.setPen(Qt.NoPen)
        p.drawEllipse(0, 0, self._size + 2, self._size + 2)
        # Core dot
        p.setBrush(self._color)
        m = 2
        p.drawEllipse(m, m, self._size - m, self._size - m)


# ── Pill badge label ──────────────────────────────────────────────────────────

class PillLabel(QLabel):
    """Small rounded-rect badge."""
    def __init__(self, text, bg, fg, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            f"background-color: {bg}; color: {fg}; border-radius: 8px; "
            f"padding: 2px 8px; font-size: 11px; font-weight: bold;"
        )


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    _log_signal = pyqtSignal(str, str)   # text, color (empty = plain)
    _svc_done   = pyqtSignal(str, bool, str)

    def __init__(self):
        super().__init__()
        self._proc          = None
        self._rviz_proc     = None
        self._log_reader    = None
        self._poll_timer    = None
        self._mode          = "mapping"
        self._svc_threads   = []

        self._log_signal.connect(self._append_colored)
        self._svc_done.connect(self._on_service_result)

        self.setWindowTitle("FASTLIO2 Control Panel")
        self.setMinimumSize(860, 720)
        self._build_ui()
        self._apply_state(running=False)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Header banner ──────────────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet(f"background-color: {C['mantle']}; border: none;")
        header.setFixedHeight(64)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(20, 0, 20, 0)
        h_lay.setSpacing(10)

        # Left side: logo + title
        lbl_title = QLabel("FASTLIO2")
        lbl_title.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {C['blue']}; "
            f"letter-spacing: 1px; background: transparent;"
        )
        lbl_sep = QLabel("·")
        lbl_sep.setStyleSheet(f"font-size: 18px; color: {C['surface2']}; background: transparent;")
        lbl_sub = QLabel("Control Panel")
        lbl_sub.setStyleSheet(f"font-size: 14px; color: {C['overlay1']}; background: transparent;")

        badge = PillLabel("ROS2", C["surface0"], C["blue"])
        badge.setFixedHeight(22)

        h_lay.addWidget(lbl_title)
        h_lay.addWidget(lbl_sep)
        h_lay.addWidget(lbl_sub)
        h_lay.addWidget(badge)
        h_lay.addStretch()

        # Right side: status
        self._status_dot = StatusDot(size=10)
        self._status_lbl = QLabel("Idle")
        self._status_lbl.setStyleSheet(
            f"color: {C['overlay1']}; font-size: 12px; font-weight: 600; background: transparent;"
        )
        h_lay.addWidget(self._status_dot)
        h_lay.addWidget(self._status_lbl)

        lay.addWidget(header)

        # ── Blue accent line ────────────────────────────────────────────────
        accent = QFrame()
        accent.setObjectName("accent_line")
        accent.setFrameShape(QFrame.HLine)
        accent.setFixedHeight(2)
        lay.addWidget(accent)

        # ── Scrollable content area ────────────────────────────────────────
        content = QWidget()
        content_lay = QVBoxLayout(content)
        content_lay.setContentsMargins(18, 14, 18, 14)
        content_lay.setSpacing(10)
        lay.addWidget(content, stretch=1)

        # ── Mode selector ──────────────────────────────────────────────────
        mode_box = QGroupBox("MODE")
        mode_lay = QHBoxLayout(mode_box)
        mode_lay.setContentsMargins(8, 4, 8, 8)
        mode_lay.setSpacing(0)

        self._btn_map = QPushButton("  🗺   Mapping  ")
        self._btn_loc = QPushButton("  📍   Localization  ")
        for btn in (self._btn_map, self._btn_loc):
            btn.setFixedHeight(38)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn_map.clicked.connect(lambda: self._set_mode("mapping"))
        self._btn_loc.clicked.connect(lambda: self._set_mode("localization"))
        mode_lay.addWidget(self._btn_map)
        mode_lay.addWidget(self._btn_loc)
        mode_lay.addStretch(1)
        content_lay.addWidget(mode_box)

        # ── Configuration ──────────────────────────────────────────────────
        cfg_box = QGroupBox("CONFIGURATION")
        cfg_lay = QVBoxLayout(cfg_box)
        cfg_lay.setContentsMargins(10, 6, 10, 10)
        cfg_lay.setSpacing(8)

        map_row = QHBoxLayout()
        map_row.setSpacing(8)
        lbl_map = QLabel("Map path")
        lbl_map.setStyleSheet(f"color: {C['subtext0']}; font-size: 12px;")
        lbl_map.setFixedWidth(68)
        self._map_edit = QLineEdit("/home/local/ISDADS/ses634/fastlio2_ws/maps/map1.pcd")
        self._map_edit.setPlaceholderText("Path to .pcd map file…")
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.setFixedWidth(82)
        self._browse_btn.setFixedHeight(32)
        self._browse_btn.clicked.connect(self._browse)
        map_row.addWidget(lbl_map)
        map_row.addWidget(self._map_edit)
        map_row.addWidget(self._browse_btn)
        cfg_lay.addLayout(map_row)

        # Data source option
        self._live_data_cb = QCheckBox("Live data  (launch Livox driver)")
        self._live_data_cb.setChecked(True)
        self._live_data_cb.setToolTip(
            "Checked: launch the Livox MID360 driver to receive live LiDAR + IMU data.\n"
            "Unchecked: skip the driver and wait for data from a rosbag instead.")
        self._live_data_cb.stateChanged.connect(self._on_live_data_changed)
        cfg_lay.addWidget(self._live_data_cb)

        # Localization-only options
        self._loc_options_widget = QWidget()
        loc_opts_lay = QHBoxLayout(self._loc_options_widget)
        loc_opts_lay.setContentsMargins(0, 2, 0, 0)
        loc_opts_lay.setSpacing(24)

        self._people_filter_cb = QCheckBox("People filter  (DBSCAN)")
        self._people_filter_cb.setChecked(True)
        self._people_filter_cb.setToolTip(
            "Launch dbscan_filter_node and route localizer through filtered cloud.\n"
            "Disable to bypass the filter and use the raw body cloud.")

        self._force_global_cb = QCheckBox("Force global search on start")
        self._force_global_cb.setChecked(True)
        self._force_global_cb.setToolTip(
            "Delete the saved pose file before launching so the localizer always\n"
            "performs a full NDT+ICP global search instead of trying the last\n"
            "known pose (warm-start).  Uncheck to keep the saved pose for a\n"
            "faster restart when the robot has not moved between sessions.")

        loc_opts_lay.addWidget(self._people_filter_cb)
        loc_opts_lay.addWidget(self._force_global_cb)
        loc_opts_lay.addStretch()
        cfg_lay.addWidget(self._loc_options_widget)
        content_lay.addWidget(cfg_box)

        # ── Launch control ────────────────────────────────────────────────
        launch_box = QGroupBox("LAUNCH")
        launch_lay = QHBoxLayout(launch_box)
        launch_lay.setContentsMargins(10, 6, 10, 10)
        launch_lay.setSpacing(10)

        self._start_btn    = self._primary_btn("▶   START",    C["green"])
        self._stop_btn     = self._outline_btn("■   STOP",     C["red"])
        self._kill_rviz_btn = self._outline_btn("✕   Kill RViz", C["peach"])

        self._start_btn.setFixedHeight(44)
        self._stop_btn.setFixedHeight(44)
        self._kill_rviz_btn.setFixedHeight(44)
        self._start_btn.setMinimumWidth(140)
        self._stop_btn.setMinimumWidth(110)
        self._kill_rviz_btn.setMinimumWidth(120)

        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        self._kill_rviz_btn.clicked.connect(self._kill_rviz)

        launch_lay.addWidget(self._start_btn)
        launch_lay.addWidget(self._stop_btn)
        launch_lay.addSpacing(8)
        launch_lay.addWidget(self._kill_rviz_btn)
        launch_lay.addStretch()
        content_lay.addWidget(launch_box)

        # ── Services ──────────────────────────────────────────────────────
        self._svc_box = QGroupBox("SERVICES")
        self._svc_grid = QGridLayout(self._svc_box)
        self._svc_grid.setContentsMargins(10, 6, 10, 10)
        self._svc_grid.setSpacing(8)
        self._svc_btns = {}
        content_lay.addWidget(self._svc_box)

        # ── Log ───────────────────────────────────────────────────────────
        log_box = QGroupBox("OUTPUT")
        log_lay = QVBoxLayout(log_box)
        log_lay.setContentsMargins(10, 6, 10, 10)
        log_lay.setSpacing(6)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(3000)
        self._log.setMinimumHeight(190)
        log_lay.addWidget(self._log)

        log_ctrl = QHBoxLayout()
        log_ctrl.addStretch()
        clr_btn = QPushButton("Clear")
        clr_btn.setFixedSize(64, 26)
        clr_btn.setStyleSheet(
            f"QPushButton {{ background: {C['surface0']}; color: {C['overlay1']}; "
            f"border: 1px solid {C['surface1']}; border-radius: 4px; font-size: 11px; }}"
            f"QPushButton:hover {{ background: {C['surface1']}; color: {C['text']}; }}"
        )
        clr_btn.clicked.connect(self._log.clear)
        log_ctrl.addWidget(clr_btn)
        log_lay.addLayout(log_ctrl)
        content_lay.addWidget(log_box, stretch=1)

        self._set_mode("mapping")

    # ── Button factories ──────────────────────────────────────────────────────

    def _primary_btn(self, text, color):
        """Solid filled primary action button."""
        btn = QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: {C["crust"]};
                font-weight: bold;
                font-size: 14px;
                border: none;
                border-radius: 7px;
                padding: 8px 28px;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
            QPushButton:pressed {{
                background-color: {color}aa;
            }}
            QPushButton:disabled {{
                background-color: {C["surface1"]};
                color: {C["overlay0"]};
            }}
        """)
        return btn

    def _outline_btn(self, text, color):
        """Outlined secondary button."""
        btn = QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {color};
                font-weight: bold;
                font-size: 13px;
                border: 1px solid {color}88;
                border-radius: 7px;
                padding: 8px 20px;
            }}
            QPushButton:hover {{
                background-color: {color}22;
                border-color: {color};
            }}
            QPushButton:pressed {{
                background-color: {color}44;
            }}
            QPushButton:disabled {{
                color: {C["surface2"]};
                border-color: {C["surface1"]};
                background-color: transparent;
            }}
        """)
        return btn

    def _service_btn(self, icon, label, color):
        """Service card button with left accent border."""
        btn = QPushButton(f"{icon}  {label}")
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {C["surface0"]};
                color: {C["text"]};
                border: 1px solid {C["surface1"]};
                border-left: 3px solid {color};
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 13px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {C["surface1"]};
                border-left-color: {color};
            }}
            QPushButton:pressed {{
                background-color: {C["surface2"]};
            }}
            QPushButton:disabled {{
                color: {C["overlay0"]};
                border-color: {C["surface0"]};
                border-left-color: {C["surface2"]};
                background-color: {C["surface0"]};
            }}
        """)
        return btn

    # ── Mode ──────────────────────────────────────────────────────────────────

    def _set_mode(self, mode):
        self._mode = mode

        def _active(is_left):
            r = ("border-top-left-radius:7px; border-bottom-left-radius:7px; "
                 "border-top-right-radius:0px; border-bottom-right-radius:0px;"
                 if is_left else
                 "border-top-left-radius:0px; border-bottom-left-radius:0px; "
                 "border-top-right-radius:7px; border-bottom-right-radius:7px;")
            return f"""
            QPushButton {{
                background-color: {C["blue"]};
                color: {C["crust"]};
                font-weight: bold;
                font-size: 13px;
                border: none;
                {r}
                padding: 0 24px;
            }}
            """

        def _inactive(is_left):
            r = ("border-top-left-radius:7px; border-bottom-left-radius:7px; "
                 "border-top-right-radius:0px; border-bottom-right-radius:0px;"
                 if is_left else
                 "border-top-left-radius:0px; border-bottom-left-radius:0px; "
                 "border-top-right-radius:7px; border-bottom-right-radius:7px;")
            return f"""
            QPushButton {{
                background-color: {C["surface0"]};
                color: {C["subtext0"]};
                font-size: 13px;
                border: 1px solid {C["surface1"]};
                {r}
                padding: 0 24px;
            }}
            QPushButton:hover {{
                background-color: {C["surface1"]};
                color: {C["text"]};
            }}
            """

        if mode == "mapping":
            self._btn_map.setStyleSheet(_active(True))
            self._btn_loc.setStyleSheet(_inactive(False))
        else:
            self._btn_map.setStyleSheet(_inactive(True))
            self._btn_loc.setStyleSheet(_active(False))

        self._loc_options_widget.setVisible(mode == "localization")
        self._rebuild_service_buttons()
        self._apply_state(self._is_running())

    def _rebuild_service_buttons(self):
        while self._svc_grid.count():
            item = self._svc_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._svc_btns.clear()

        svcs = SERVICES.get(self._mode, [])
        cols = 2
        for i, svc in enumerate(svcs):
            btn = self._service_btn(svc["icon"], svc["label"], svc["color"])
            btn.setFixedHeight(44)
            sid = svc["id"]
            btn.clicked.connect(lambda _checked, s=svc: self._call_service(s))
            self._svc_grid.addWidget(btn, i // cols, i % cols)
            self._svc_btns[sid] = btn

    def _on_live_data_changed(self, _state):
        """Update dependent widget states when the live-data checkbox changes."""
        self._apply_state(self._is_running())

    # ── Browse ────────────────────────────────────────────────────────────────

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PCD map", str(Path.home()), "PCD files (*.pcd);;All (*)"
        )
        if path:
            self._map_edit.setText(path)

    # ── Launch ────────────────────────────────────────────────────────────────

    def _rviz_config_path(self):
        """Resolve the RViz config path for the current mode via ros2 pkg prefix."""
        pkg = "fastlio2" if self._mode == "mapping" else "localizer"
        cfg = "fastlio2.rviz" if self._mode == "mapping" else "localizer.rviz"
        try:
            prefix = subprocess.check_output(
                ["ros2", "pkg", "prefix", pkg], text=True,
                env={**os.environ}, timeout=5,
            ).strip()
            return os.path.join(prefix, "share", pkg, "rviz", cfg)
        except Exception:
            return None

    def _kill_rviz(self):
        if self._rviz_proc and self._rviz_proc.poll() is None:
            self._log_colored("─── killing RViz ───", C["peach"])
            try:
                self._rviz_proc.terminate()
            except ProcessLookupError:
                pass
            self._rviz_proc = None
        else:
            self._log_colored("RViz is not running.", C["overlay1"])
        self._update_rviz_btn()

    def _update_rviz_btn(self):
        alive = self._rviz_proc is not None and self._rviz_proc.poll() is None
        self._kill_rviz_btn.setEnabled(alive)

    def _launch_rviz(self):
        """Start RViz as an independent process (survives stack restarts)."""
        if self._rviz_proc and self._rviz_proc.poll() is None:
            self._log_colored("RViz already running — reusing existing window", C["overlay1"])
            return
        cfg = self._rviz_config_path()
        cmd = ["rviz2"] + (["-d", cfg] if cfg else [])
        self._log_colored(f"$ {' '.join(cmd)}", C["blue"])
        try:
            self._rviz_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env={**os.environ},
            )
        except FileNotFoundError as exc:
            self._log_colored(f"[ERROR] Could not launch RViz: {exc}", C["red"])

    def _build_cmd(self):
        map_path  = self._map_edit.text().strip()
        live_data = self._live_data_cb.isChecked()

        if self._mode == "mapping":
            launch_file = "mapping_full.launch.py" if live_data else "mapping.launch.py"
            return ["ros2", "launch", "fastlio2_bringup", launch_file, "launch_rviz:=false"]

        # localization
        force_global = "true" if self._force_global_cb.isChecked() else "false"
        if live_data:
            use_filter = "true" if self._people_filter_cb.isChecked() else "false"
            return [
                "ros2", "launch", "fastlio2_bringup", "localization_full.launch.py",
                f"map_path:={map_path}",
                f"use_people_filter:={use_filter}",
                f"force_global_search:={force_global}",
                "launch_rviz:=false",
            ]
        return [
            "ros2", "launch", "fastlio2_bringup", "localization.launch.py",
            f"map_path:={map_path}",
            f"force_global_search:={force_global}",
            "launch_rviz:=false",
        ]

    def _start(self):
        if self._is_running():
            return
        cmd = self._build_cmd()
        self._log_colored(f"$ {' '.join(cmd)}", C["blue"])
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,   # own process group → killpg works
                env={**os.environ},
            )
        except FileNotFoundError as exc:
            self._log_colored(f"[ERROR] {exc}", C["red"])
            return
        self._launch_rviz()
        self._update_rviz_btn()
        self._log_reader = LogReader(self._proc)
        self._log_reader.line.connect(lambda ln: self._log_signal.emit(ln, ""))
        self._log_reader.start()
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._check_proc)
        self._poll_timer.start(1000)
        self._apply_state(running=True)

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._log_colored("─── sending SIGTERM to process group ───", C["yellow"])
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            QTimer.singleShot(4000, self._force_kill)

    def _force_kill(self):
        if self._proc and self._proc.poll() is None:
            self._log_colored("─── force killing process group ───", C["red"])
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _check_proc(self):
        if self._proc and self._proc.poll() is not None:
            self._poll_timer.stop()
            self._apply_state(running=False)
            self._log_colored(
                f"─── process exited (code {self._proc.returncode}) ───", C["peach"]
            )

    def _is_running(self):
        return self._proc is not None and self._proc.poll() is None

    # ── State ─────────────────────────────────────────────────────────────────

    def _apply_state(self, running):
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._btn_map.setEnabled(not running)
        self._btn_loc.setEnabled(not running)
        self._map_edit.setEnabled(not running)
        self._browse_btn.setEnabled(not running)
        self._live_data_cb.setEnabled(not running)
        live = self._live_data_cb.isChecked()
        self._people_filter_cb.setEnabled(not running and live)
        self._force_global_cb.setEnabled(not running)
        self._update_rviz_btn()
        for btn in self._svc_btns.values():
            btn.setEnabled(running)
        if running:
            mode_str = "Mapping" if self._mode == "mapping" else "Localization"
            self._status_dot.set_color(C["green"])
            self._status_lbl.setText(f"Running  ·  {mode_str}")
            self._status_lbl.setStyleSheet(
                f"color: {C['green']}; font-size: 12px; font-weight: 600; background: transparent;"
            )
        else:
            self._status_dot.set_color(C["surface2"])
            self._status_lbl.setText("Idle")
            self._status_lbl.setStyleSheet(
                f"color: {C['overlay1']}; font-size: 12px; font-weight: 600; background: transparent;"
            )

    # ── Services ──────────────────────────────────────────────────────────────

    def _call_service(self, svc):
        map_path = self._map_edit.text().strip()
        args_str = svc["args"](map_path)
        cmd = ["ros2", "service", "call", svc["srv"], svc["iface"], args_str]
        self._log_colored(f"$ {' '.join(cmd)}", C["sapphire"])
        btn = self._svc_btns.get(svc["id"])
        if btn:
            btn.setEnabled(False)
            btn.setText(f"⏳  {svc['label']}…")
        t = ServiceThread(
            svc["label"], cmd, svc["srv"],
            timeout=svc.get("timeout", 20),
            parse_response=svc.get("parse_response"),
        )
        t.result.connect(self._svc_done)
        self._svc_threads.append(t)
        t.start()

    def _on_service_result(self, label, ok, message):
        icon  = "✓" if ok else "✗"
        color = C["green"] if ok else C["red"]
        self._log_colored(f"{icon} [{label}]  {message}", color)
        running = self._is_running()
        for sid, btn in self._svc_btns.items():
            for svc in SERVICES.get(self._mode, []):
                if svc["id"] == sid:
                    btn.setText(f"{svc['icon']}  {svc['label']}")
                    btn.setEnabled(running)
                    break

    # ── Log ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _ts():
        """Current time as a dim HH:MM:SS prefix."""
        return datetime.now().strftime("%H:%M:%S")

    def _append_colored(self, text, color):
        """Slot for log_signal — runs on main thread."""
        if color:
            self._log_colored(text, color)
        else:
            self._log_plain(text)

    def _log_plain(self, text):
        ts  = self._ts()
        esc = html.escape(text)
        low = text.lower()
        ts_span = f'<span style="color:{C["surface2"]};">{ts}</span>  '
        if any(x in low for x in ("error", "fatal", "exception", "terminate")):
            self._log.appendHtml(f'{ts_span}<span style="color:{C["red"]};">{esc}</span>')
        elif "warn" in low:
            self._log.appendHtml(f'{ts_span}<span style="color:{C["yellow"]};">{esc}</span>')
        elif "─── process" in text:
            self._log.appendHtml(f'{ts_span}<span style="color:{C["peach"]};">{esc}</span>')
        else:
            self._log.appendHtml(f'{ts_span}<span style="color:{C["subtext1"]};">{esc}</span>')
        self._log.moveCursor(QTextCursor.End)

    def _log_colored(self, text, color):
        ts      = self._ts()
        ts_span = f'<span style="color:{C["surface2"]};">{ts}</span>  '
        self._log.appendHtml(
            f'{ts_span}<span style="color:{color};">{html.escape(text)}</span>'
        )
        self._log.moveCursor(QTextCursor.End)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._is_running():
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        # RViz left alive intentionally
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
