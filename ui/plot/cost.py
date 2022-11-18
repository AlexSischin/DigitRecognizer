import numpy as np
import pyqtgraph as pg

from ui.metrics_dispatcher import TrainMetric
from ui.plot.base_plot import BasePlot


class CostPlot(BasePlot):
    def __init__(self) -> None:
        super().__init__()

        self.setStyleSheet("border: 1px solid black;")
        self.setTitle("Cost by train data")

        self.setLabel('left', "Cost")
        self.setLabel('bottom', "Train data")
        self.curve = self.plot(pen=(0, 128, 0), symbol='t', symbolBrush=(0, 128, 0))
        self.data = np.random.normal(size=(10, 1000))
        self.showGrid(x=True, y=True)

        self.lr = pg.LinearRegionItem((0, 0))
        self.lr.setZValue(-10)
        self.lr.hide()
        self.addItem(self.lr)

    def set_data(self, metrics: list[TrainMetric]):
        nodes = [(m.data_used, m.cost) for m in metrics]
        self.curve.setData(np.array(nodes))
        if metrics:
            self.lr.setBounds((metrics[0].data_used, metrics[-1].data_used))
            if not self.lr.isVisible():
                self.lr.show()
                self.lr.setRegion((metrics[0].data_used, metrics[min(5, len(metrics) - 1)].data_used))
        else:
            self.lr.hide()
