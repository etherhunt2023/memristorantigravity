"""Main application window for the COMSOL2Neuromorphic GUI.

Implements a professional tabbed interface with a dark-themed stylesheet,
menu bar, status bar, and all simulation/analysis tabs.
"""

import sys

from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.compact_model_tab import CompactModelTab
from gui.widgets.comsol_tab import COMSOLTab
from gui.widgets.crossbar_tab import CrossbarTab
from gui.widgets.device_tab import DeviceTab
from gui.widgets.extraction_tab import ExtractionTab
from gui.widgets.fitting_tab import FittingTab
from gui.widgets.snn_tab import SNNTab

# Professional dark theme stylesheet
DARK_STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
    font-size: 10pt;
}

QTabWidget::pane {
    border: 1px solid #45475a;
    background-color: #1e1e2e;
    border-radius: 4px;
}

QTabWidget::tab-bar {
    alignment: center;
}

QTabBar::tab {
    background-color: #313244;
    color: #bac2de;
    padding: 8px 18px;
    margin: 2px 1px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border: 1px solid #45475a;
    border-bottom: none;
    min-width: 100px;
}

QTabBar::tab:selected {
    background-color: #45475a;
    color: #cba6f7;
    font-weight: bold;
    border-bottom: 2px solid #cba6f7;
}

QTabBar::tab:hover:!selected {
    background-color: #3b3d52;
    color: #f5c2e7;
}

QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #89b4fa;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    background-color: #313244;
    border-radius: 4px;
}

QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
    color: #89b4fa;
}

QPushButton:pressed {
    background-color: #585b70;
}

QPushButton:disabled {
    background-color: #1e1e2e;
    color: #585b70;
    border-color: #313244;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #89b4fa;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #45475a;
    selection-color: #cba6f7;
    border: 1px solid #585b70;
}

QTableWidget {
    background-color: #181825;
    alternate-background-color: #1e1e2e;
    color: #cdd6f4;
    gridline-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    selection-background-color: #45475a;
    selection-color: #f5c2e7;
}

QTableWidget QHeaderView::section {
    background-color: #313244;
    color: #89b4fa;
    border: 1px solid #45475a;
    padding: 4px 8px;
    font-weight: bold;
}

QTextEdit {
    background-color: #181825;
    color: #a6adc8;
    border: 1px solid #45475a;
    border-radius: 4px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 9pt;
}

QProgressBar {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    text-align: center;
    color: #cdd6f4;
    height: 20px;
}

QProgressBar::chunk {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #89b4fa, stop:1 #cba6f7
    );
    border-radius: 5px;
}

QCheckBox {
    spacing: 8px;
    color: #cdd6f4;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #585b70;
    background-color: #313244;
}

QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

QLabel {
    color: #bac2de;
}

QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
}

QMenuBar::item:selected {
    background-color: #313244;
    color: #cba6f7;
}

QMenu {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #45475a;
}

QMenu::item:selected {
    background-color: #45475a;
    color: #cba6f7;
}

QStatusBar {
    background-color: #181825;
    color: #6c7086;
    border-top: 1px solid #313244;
    font-size: 9pt;
}

QSplitter::handle {
    background-color: #45475a;
    height: 2px;
}

QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #585b70;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #89b4fa;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #1e1e2e;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background-color: #585b70;
    border-radius: 5px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #89b4fa;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""

# Tab icon emoji labels (used as text since PySide6 supports unicode)
TAB_ICONS = {
    "COMSOL Import": "📂",
    "Parameter Extraction": "📊",
    "Compact Models": "⚡",
    "Model Fitting": "🔧",
    "Device Simulation": "🔬",
    "Crossbar Array": "🧮",
    "SNN Simulation": "🧠",
}


class MainWindow(QMainWindow):
    """Main application window with tabbed interface."""

    def __init__(self) -> None:
        """Initializes the MainWindow."""
        super().__init__()
        self.setWindowTitle("COMSOL2Neuromorphic — Memristor Neuromorphic Toolkit")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        self._init_ui()
        self._init_menu_bar()
        self._init_status_bar()

    def _init_ui(self) -> None:
        """Builds the main window layout with tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(6, 6, 6, 6)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)

        # Create tab instances
        tabs_config = [
            ("📂  COMSOL Import", COMSOLTab),
            ("📊  Extraction", ExtractionTab),
            ("⚡  Compact Models", CompactModelTab),
            ("🔧  Model Fitting", FittingTab),
            ("🔬  Device Sim", DeviceTab),
            ("🧮  Crossbar", CrossbarTab),
            ("🧠  SNN", SNNTab),
        ]

        for label, tab_class in tabs_config:
            tab_widget = tab_class(self)
            self.tabs.addTab(tab_widget, label)

        layout.addWidget(self.tabs)

    def _init_menu_bar(self) -> None:
        """Builds the application menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open COMSOL File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_comsol_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _init_status_bar(self) -> None:
        """Initializes the status bar."""
        status_bar = QStatusBar()
        status_bar.showMessage("COMSOL2Neuromorphic v0.1.0 — Ready")
        self.setStatusBar(status_bar)

    def _open_comsol_file(self) -> None:
        """Opens a COMSOL file via the COMSOL Import tab."""
        self.tabs.setCurrentIndex(0)
        comsol_tab = self.tabs.widget(0)
        if hasattr(comsol_tab, "_browse_file"):
            comsol_tab._browse_file()

    def _show_about(self) -> None:
        """Shows the About dialog."""
        QMessageBox.about(
            self,
            "About COMSOL2Neuromorphic",
            "<h3>COMSOL2Neuromorphic</h3>"
            "<p>Version 0.1.0</p>"
            "<p>An open-source, research-grade Python framework for converting "
            "COMSOL memristor simulations into hardware-aware neuromorphic "
            "network architectures.</p>"
            "<p><b>Modules:</b></p>"
            "<ul>"
            "<li>COMSOL Parser & Data Import</li>"
            "<li>Electrical Parameter Extraction</li>"
            "<li>Physics-based Compact Models (VTEAM, Yakopcic, Simmons)</li>"
            "<li>Two-stage Parameter Fitting Engine</li>"
            "<li>Hardware-Aware Memristor Device (D2D, C2C, Drift, Noise)</li>"
            "<li>Crossbar Array MNA Simulator</li>"
            "<li>Spiking Neural Network (SNN) Simulator</li>"
            "</ul>"
            "<p>Built with PySide6, NumPy, SciPy, PyTorch, and Matplotlib.</p>",
        )


def main() -> None:
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Consistent cross-platform base style
    app.setStyleSheet(DARK_STYLESHEET)

    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
