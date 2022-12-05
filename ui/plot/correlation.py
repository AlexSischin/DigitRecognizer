import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import PlotWidget

import ai
from ui.metrics_dispatcher import Hub


class CorrelationHub(Hub):
    def __init__(self) -> None:
        super().__init__()
        self._act_outputs = []
        self._exp_outputs = []

    def update_data(self, metrics: list[ai.TrainMetric]):
        outputs = [np.concatenate(m.outputs, axis=None) for m in metrics]
        expected = [np.concatenate(m.expected, axis=None) for m in metrics]
        self._act_outputs.extend(outputs)
        self._exp_outputs.extend(expected)

    def calc(self, left, right):
        act_outputs = np.concatenate(self._act_outputs[left:right])
        exp_outputs = np.concatenate(self._exp_outputs[left:right]).flatten()
        return act_outputs, exp_outputs


class CorrelationWidget(QWidget):
    def __init__(self, hub: CorrelationHub) -> None:
        super().__init__()

        self._hub = hub

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        plot = PlotWidget()
        plot.setTitle("Actual activation by expected")
        plot.setLabel('left', "Actual activation")
        plot.setLabel('bottom', "Expected activation")
        self._scatter = pg.ScatterPlotItem([], [], pen=None, symbol='t1', symbolPen=None,
                                           symbolSize=10, symbolBrush=(100, 100, 255, 50))
        plot.addItem(self._scatter)
        plot.plot([0, 1])

        layout.addWidget(plot)
        self.setLayout(layout)

    def update_data(self, left, right):
        if left is not None and right is not None:
            act_outputs_vec, exp_outputs_vec = self._hub.calc(left, right)
        else:
            act_outputs_vec, exp_outputs_vec = [], []
        self._scatter.setData(exp_outputs_vec, act_outputs_vec)
