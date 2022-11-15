import pyqtgraph as pg

import ai
from ui.plot.base_plot import BasePlot
from utils import zip_utils as zp


class CorrelationPlot(BasePlot):
    def __init__(self) -> None:
        super().__init__()

        self.setStyleSheet("border: 1px solid black;")
        self.setTitle("Actual activation by expected")
        self.setLabel('left', "Actual activation")
        self.setLabel('bottom', "Expected activation")
        self._scatter = pg.ScatterPlotItem([], [], pen=None, symbol='t1', symbolPen=None,
                                           symbolSize=10, symbolBrush=(100, 100, 255, 50))
        self.addItem(self._scatter)
        self.plot([0, 1])

    def set_data(self, metrics: list[ai.TrainMetric]):
        outputs = []
        exp_outputs = []
        for metric in metrics:
            for output, expected in zp.zip2(metric.outputs, metric.expected):
                outputs.extend(output)
                exp_outputs.extend(expected)
        self._scatter.setData(exp_outputs, outputs)

