import numpy as np
import pyqtgraph as pg

import ai
from ui.plot.base_plot import BasePlot


def square_mean_costs(costs: list[np.ndarray]) -> np.ndarray:
    costs_array = np.array(costs)
    squared_costs = np.square(costs_array)
    return np.mean(squared_costs, axis=0)


class CostPlot(BasePlot):
    def __init__(self, train_data_chunk_size: int) -> None:
        super().__init__()
        self._train_data_chunk_size = train_data_chunk_size

        self.setStyleSheet("border: 1px solid black;")
        self.setTitle("Cost by train data")

        self.setLabel('left', "Cost")
        self.setLabel('bottom', "Train data")
        self.curve = self.plot(pen=(0, 128, 0), symbol='t', symbolBrush=(0, 128, 0))
        self.data = np.random.normal(size=(10, 1000))
        self.showGrid(x=True, y=True)

        self.lr = pg.LinearRegionItem([0, 5 * self._train_data_chunk_size])
        self.lr.setZValue(-10)
        self.lr.hide()
        self.addItem(self.lr)

    def set_data(self, data_used: list[int], metrics: list[ai.TrainMetric]):
        costs = [sum(square_mean_costs(m.costs)) for m in metrics]
        nodes = np.array([data_used, costs]).transpose()
        self.curve.setData(nodes)
        if data_used:
            self.lr.setBounds((0, data_used[-1]))
            self.lr.show()
        else:
            self.lr.hide()
