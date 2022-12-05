import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import PlotWidget

import ai
from ui.metrics_dispatcher import Hub


class BGradHub(Hub):
    def __init__(self) -> None:
        super().__init__()
        self._bs = []
        self._bgs = []

    def update_data(self, metrics: list[ai.TrainMetric]):
        for m in metrics:
            self._bs.append(m.b)
            self._bgs.append(m.b_gradient)

    def calc(self, left, right, layer):
        l_grad = self._bgs[left][layer - 1]
        l_state = self._bs[left][layer - 1]
        r_state = self._bs[right - 1][layer - 1]
        grad1d = l_grad + l_state - r_state
        grad = np.expand_dims(grad1d, axis=1)
        return grad


class BGradWidget(QWidget):
    def __init__(self, hub: BGradHub):
        super().__init__()

        self._hub = hub

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        plot = PlotWidget()
        self._item = plot.getPlotItem()
        plot.setTitle("Bias gradient")
        plot.setLabel('left', "Left layer")
        plot.setLabel('bottom', "Right layer")

        cm = pg.colormap.get('CET-D4')
        self._bar = pg.ColorBarItem(values=(-10 ** -1, 10 ** -1), rounding=10 ** -9, colorMap=cm)
        self._img = pg.ImageItem()
        self._item.addItem(self._img)

        layout.addWidget(plot)
        self.setLayout(layout)

    def update_data(self, left, right, layer):
        if left is not None and right is not None and layer is not None and layer > 0:
            grad = self._hub.calc(left, right, layer)
            self._img.setImage(image=grad)
        else:
            self._img.clear()
        self._bar.setImageItem(self._img, insert_in=self._item)
