"""COMSOL file parsing and data preview tab widget.

Allows the user to browse for COMSOL export files, parse them using the
COMSOLParser, and view the resulting data in a table and metadata panel.
"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class COMSOLTab(QWidget):
    """Tab widget for importing and previewing COMSOL simulation export files."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initializes the COMSOLTab.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.dataset = None
        self._init_ui()

    def _init_ui(self) -> None:
        """Builds the tab user interface layout."""
        layout = QVBoxLayout(self)

        # --- File selection group ---
        file_group = QGroupBox("COMSOL File Import")
        file_layout = QHBoxLayout(file_group)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a COMSOL export file (.txt, .csv, .dat)...")
        self.file_path_edit.setReadOnly(True)

        browse_btn = QPushButton("Browse...")
        browse_btn.setMinimumWidth(100)
        browse_btn.clicked.connect(self._browse_file)

        parse_btn = QPushButton("Parse File")
        parse_btn.setMinimumWidth(100)
        parse_btn.setObjectName("parseButton")
        parse_btn.clicked.connect(self._parse_file)

        file_layout.addWidget(QLabel("File:"))
        file_layout.addWidget(self.file_path_edit, 1)
        file_layout.addWidget(browse_btn)
        file_layout.addWidget(parse_btn)

        layout.addWidget(file_group)

        # --- Data preview splitter ---
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Data table
        table_group = QGroupBox("Parsed Data Preview")
        table_layout = QVBoxLayout(table_group)
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.row_count_label = QLabel("Rows: 0 | Columns: 0")
        table_layout.addWidget(self.data_table)
        table_layout.addWidget(self.row_count_label)
        splitter.addWidget(table_group)

        # Metadata panel
        meta_group = QGroupBox("File Metadata")
        meta_layout = QVBoxLayout(meta_group)
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setMaximumHeight(200)
        meta_layout.addWidget(self.metadata_text)
        splitter.addWidget(meta_group)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

        # Status bar
        self.status_label = QLabel("Ready. Select a COMSOL export file to begin.")
        layout.addWidget(self.status_label)

    def _browse_file(self) -> None:
        """Opens a file dialog to select a COMSOL export file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select COMSOL Export File",
            "",
            "COMSOL Files (*.txt *.csv *.dat);;All Files (*)",
        )
        if file_path:
            self.file_path_edit.setText(file_path)

    def _parse_file(self) -> None:
        """Parses the selected file using COMSOLParser and populates the UI."""
        file_path = self.file_path_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "No File Selected", "Please select a file first.")
            return

        path = Path(file_path)
        if not path.exists():
            QMessageBox.critical(self, "File Not Found", f"File does not exist:\n{path}")
            return

        try:
            from comsol.parser import COMSOLParser

            parser = COMSOLParser()
            self.dataset = parser.parse(path)

            # Populate table
            df = self.dataset.data
            self.data_table.setRowCount(min(len(df), 500))  # Cap preview at 500 rows
            self.data_table.setColumnCount(len(df.columns))
            self.data_table.setHorizontalHeaderLabels([str(c) for c in df.columns])

            for row_idx in range(min(len(df), 500)):
                for col_idx in range(len(df.columns)):
                    val = df.iloc[row_idx, col_idx]
                    item = QTableWidgetItem(f"{val:.6g}" if isinstance(val, float) else str(val))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.data_table.setItem(row_idx, col_idx, item)

            self.row_count_label.setText(
                f"Rows: {len(df)} | Columns: {len(df.columns)}"
                + (" (showing first 500)" if len(df) > 500 else "")
            )

            # Populate metadata
            meta_lines = []
            for key, value in self.dataset.metadata.items():
                if key == "comments":
                    meta_lines.append(f"Comments ({len(value)} lines)")
                else:
                    meta_lines.append(f"{key}: {value}")
            self.metadata_text.setPlainText("\n".join(meta_lines))

            self.status_label.setText(
                f"✓ Parsed successfully: {len(df)} rows, {len(df.columns)} columns"
            )

        except Exception as e:
            QMessageBox.critical(self, "Parsing Error", f"Failed to parse file:\n{e}")
            self.status_label.setText(f"✗ Error: {e}")

    def get_dataset(self):
        """Returns the currently parsed COMSOLDataset, or None.

        Returns:
            COMSOLDataset | None: The parsed dataset.
        """
        return self.dataset
