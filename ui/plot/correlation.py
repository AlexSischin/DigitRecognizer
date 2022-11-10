import pyqtgraph as pg

import ai
from utils import zip_utils as zp


class CorrelationPlot:
    def __init__(self, plot) -> None:
        super().__init__()
        self._plot = plot

        self._plot.setTitle("Actual activation by expected")
        self._plot.setLabel('left', "Actual activation")
        self._plot.setLabel('bottom', "Expected activation")
        self._scatter = pg.ScatterPlotItem([], [], pen=None, symbol='t1', symbolPen=None,
                                           symbolSize=10, symbolBrush=(100, 100, 255, 50))
        self._plot.addItem(self._scatter)
        self._plot.plot([0, 1])

    def update(self, metrics):
        outputs = []
        exp_outputs = []
        for metric in metrics:
            for output, expected in zp.zip2(metric.outputs, metric.expected):
                outputs.extend(output)
                exp_outputs.extend(expected)
        self._scatter.setData(exp_outputs, outputs)

