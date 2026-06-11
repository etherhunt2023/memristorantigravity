"""Entry point for the COMSOL2Neuromorphic GUI application.

Launches the PySide6-based Windows GUI with all simulation and analysis tabs.

Usage:
    python examples/launch_gui.py
"""

from gui.main_window import main

if __name__ == "__main__":
    main()
