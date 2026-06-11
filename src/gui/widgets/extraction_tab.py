"""Parameter extraction tab widget.

Provides controls to extract memristor electrical parameters (LRS, HRS, Vset,
Vreset, non-linearity) from a parsed COMSOL dataset and display the results.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ExtractionTab(QWidget):
    """Tab widget for extracting electrical parameters from COMSOL data."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initializes the ExtractionTab.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.extracted_params = None
        self._init_ui()

    def _init_ui(self) -> None:
        """Builds the extraction tab UI."""
        layout = QVBoxLayout(self)

        # --- Configuration group ---
        config_group = QGroupBox("Extraction Configuration")
        config_layout = QFormLayout(config_group)

        self.lrs_voltage_spin = QDoubleSpinBox()
        self.lrs_voltage_spin.setRange(0.001, 5.0)
        self.lrs_voltage_spin.setValue(0.1)
        self.lrs_voltage_spin.setDecimals(3)
        self.lrs_voltage_spin.setSuffix(" V")
        config_layout.addRow("LRS Read Voltage:", self.lrs_voltage_spin)

        self.hrs_voltage_spin = QDoubleSpinBox()
        self.hrs_voltage_spin.setRange(0.001, 5.0)
        self.hrs_voltage_spin.setValue(0.1)
        self.hrs_voltage_spin.setDecimals(3)
        self.hrs_voltage_spin.setSuffix(" V")
        config_layout.addRow("HRS Read Voltage:", self.hrs_voltage_spin)

        self.nl_voltage_spin = QDoubleSpinBox()
        self.nl_voltage_spin.setRange(0.01, 10.0)
        self.nl_voltage_spin.setValue(1.0)
        self.nl_voltage_spin.setDecimals(2)
        self.nl_voltage_spin.setSuffix(" V")
        config_layout.addRow("Non-linearity Voltage:", self.nl_voltage_spin)

        layout.addWidget(config_group)

        # --- Action button ---
        btn_layout = QHBoxLayout()
        self.extract_btn = QPushButton("Extract Parameters")
        self.extract_btn.setObjectName("extractButton")
        self.extract_btn.setMinimumHeight(36)
        self.extract_btn.clicked.connect(self._run_extraction)
        btn_layout.addStretch()
        btn_layout.addWidget(self.extract_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # --- Results table ---
        results_group = QGroupBox("Extracted Parameters")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value", "Unit"])
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)

        layout.addWidget(results_group, 1)

        # Status
        self.status_label = QLabel("Ready. Parse a COMSOL file first, then extract parameters.")
        layout.addWidget(self.status_label)

    def _run_extraction(self) -> None:
        """Runs the extraction pipeline using the dataset from the COMSOL tab."""
        # Find the COMSOL tab via parent main window
        main_window = self.window()
        comsol_tab = None
        if hasattr(main_window, "tabs"):
            for i in range(main_window.tabs.count()):
                widget = main_window.tabs.widget(i)
                if hasattr(widget, "get_dataset"):
                    comsol_tab = widget
                    break

        if comsol_tab is None or comsol_tab.get_dataset() is None:
            QMessageBox.warning(
                self,
                "No Dataset",
                "Please parse a COMSOL file in the 'COMSOL Import' tab first.",
            )
            return

        dataset = comsol_tab.get_dataset()

        try:
            from extraction.extractor import ElectricalParameterExtractor
            from utils.config_loader import Config

            config = Config(
                {
                    "extraction": {
                        "lrs_read_voltage": self.lrs_voltage_spin.value(),
                        "hrs_read_voltage": self.hrs_voltage_spin.value(),
                        "non_linearity_voltage": self.nl_voltage_spin.value(),
                    }
                }
            )

            extractor = ElectricalParameterExtractor(config=config)
            self.extracted_params = extractor.extract(dataset)

            # Populate results table
            params_dict = self.extracted_params.to_dict()
            units = {
                "r_lrs": "Ω",
                "r_hrs": "Ω",
                "r_ratio": "—",
                "v_set": "V",
                "v_reset": "V",
                "non_linearity": "—",
                "switching_time": "s",
            }

            self.results_table.setRowCount(len(params_dict))
            for row_idx, (key, value) in enumerate(params_dict.items()):
                self.results_table.setItem(row_idx, 0, QTableWidgetItem(key))
                val_str = f"{value:.4g}" if value is not None else "N/A"
                item = QTableWidgetItem(val_str)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.results_table.setItem(row_idx, 1, item)
                self.results_table.setItem(row_idx, 2, QTableWidgetItem(units.get(key, "")))

            self.status_label.setText("✓ Parameters extracted successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Extraction Error", f"Extraction failed:\n{e}")
            self.status_label.setText(f"✗ Error: {e}")

    def get_extracted_params(self):
        """Returns the extracted parameters, or None.

        Returns:
            ExtractedParameters | None: The extraction result.
        """
        return self.extracted_params
