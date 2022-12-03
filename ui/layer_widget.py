from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QWidget, QLabel, QVBoxLayout


class LayerWidget(QWidget):
    sigLayerSelected = pyqtSignal(int)

    def __init__(self, layer_count=0) -> None:
        super().__init__()
        self._layer_count = None

        self._init_layout()

        self.set_layer_count(layer_count)

    def _init_layout(self):
        layout = QVBoxLayout()

        self._label = QLabel('Layer:')
        layout.addWidget(self._label)

        self._combo_box = QComboBox()
        self._combo_box.currentTextChanged.connect(self._on_text_changed)
        layout.addWidget(self._combo_box)

        self.setLayout(layout)

    def set_layer_count(self, n):
        if n == self._layer_count:
            return
        self._layer_count = n
        self._combo_box.clear()
        for layer in range(n):
            self._combo_box.addItem(f'{layer}')
        if n > 0:
            self._combo_box.setCurrentIndex(n - 1)

    def _on_text_changed(self, layer_text):
        self.sigLayerSelected.emit(int(layer_text))
