import numpy as np

import ai
from ui.plot.base_plot import BasePlot


class GradientLengthPlot(BasePlot):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet("border: 1px solid black;")
        self.setTitle("Gradient length by data")

        self.setLabel('left', "Gradient length")
        self.setLabel('bottom', "Train data")
        self.curve = self.plot(pen=(250, 194, 5), symbol='t', symbolBrush=(250, 194, 5))
        self.data = np.random.normal(size=(10, 1000))
        self.showGrid(x=True, y=True)

    def set_data(self, data_used: list[int], metrics: list[ai.TrainMetric]):
        g_lengths = [m.gradient_len for m in metrics]
        nodes = np.array([data_used, g_lengths]).transpose()
        self.curve.setData(nodes)
