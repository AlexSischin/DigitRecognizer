from PyQt5 import QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QMainWindow

import ai
from ui.test.test_widget import TestWidget


class TestWindow(QMainWindow):
    sigClosed = pyqtSignal(object)

    def __init__(self, ai_version, ai_model: ai.Ai, title, test_data):
        super().__init__()
        self.ai_version = ai_version
        self._ai_model = ai_model
        self._test_data = test_data

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        self._central_widget = TestWidget(self, self._ai_model, self._test_data)
        self.setCentralWidget(self._central_widget)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        self.sigClosed.emit(self)
