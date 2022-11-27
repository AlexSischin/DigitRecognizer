from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTabWidget

from ui.test.dataset.test_data import DatasetTestWidget


class TestWidget(QWidget):
    def __init__(self, parent, ai_model, test_data) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout()
        self._tab_widget = QTabWidget()

        self._tab_widget.addTab(DatasetTestWidget(ai_model, test_data), 'Dataset')
        self._tab_widget.addTab(QLabel('Draw'), 'Draw')

        self._layout.addWidget(self._tab_widget)
        self.setLayout(self._layout)
