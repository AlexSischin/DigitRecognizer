import numpy as np

from ui.metrics_dispatcher import TrainMetric
from ui.plot.base_plot import BasePlot


class GradientLengthPlot(BasePlot):
    def __init__(self):
        super().__init__()

        self.setStyleSheet("border: 1px solid black;")
        self.setTitle("Gradient length by data")

        self.setLabel('left', "Gradient length")
        self.setLabel('bottom', "Train data")
        self.curve = self.plot(pen=(250, 194, 5), symbol='t', symbolBrush=(250, 194, 5))
        self.data = np.random.normal(size=(10, 1000))
        self.showGrid(x=True, y=True)

    def set_data(self, metrics: list[TrainMetric]):
        nodes = [(m.data_used, m.gradient_len) for m in metrics]
        self.curve.setData(np.array(nodes))
