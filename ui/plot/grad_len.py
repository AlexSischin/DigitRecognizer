import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import PlotWidget

import ai
from ui.metrics_dispatcher import Hub


class GradLenHub(Hub):
    def __init__(self) -> None:
        super().__init__()
        self._grad_lens = []

    def update_data(self, metrics: list[ai.TrainMetric]):
        self._grad_lens.extend([(m.data_used, m.gradient_len) for m in metrics])

    def calc(self, left, right):
        return np.array(self._grad_lens[left:right])


class GradLenWidget(QWidget):
    def __init__(self, hub: GradLenHub):
        super().__init__()

        self._hub = hub

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        plot = PlotWidget()
        plot.setTitle("Gradient length by data")

        plot.setLabel('left', "Gradient length")
        plot.setLabel('bottom', "Train data")
        self._curve = plot.plot(pen=(250, 194, 5), symbol='t', symbolBrush=(250, 194, 5))
        plot.showGrid(x=True, y=True)

        layout.addWidget(plot)
        self.setLayout(layout)

    def update_data(self, left, right):
        if left is not None and right is not None:
            nodes = self._hub.calc(left, right)
        else:
            nodes = np.array([])
        self._curve.setData(nodes)
