import numpy as np
import pyqtgraph as pg

import ai


def square_mean_costs(costs: list[np.ndarray]) -> np.ndarray:
    costs_array = np.array(costs)
    squared_costs = np.square(costs_array)
    return np.mean(squared_costs, axis=0)


class CostPlot:
    def __init__(self, plot, train_data_chunk_size: int) -> None:
        super().__init__()
        self._plot = plot
        self._train_data_chunk_size = train_data_chunk_size

        self._plot.setTitle("Cost by train data")

        self._plot.setLabel('left', "Cost")
        self._plot.setLabel('bottom', "Train data")
        self.curve = self._plot.plot(pen=(0, 128, 0), symbol='t', symbolBrush=(0, 128, 0))
        self.data = np.random.normal(size=(10, 1000))
        self._plot.showGrid(x=True, y=True)

        self.lr = pg.LinearRegionItem([0, 5 * self._train_data_chunk_size])
        self.lr.setZValue(-10)
        self._plot.addItem(self.lr)

    def update(self, data_used: list[int], metrics: list[ai.TrainMetric]):
        costs = [sum(square_mean_costs(m.costs)) for m in metrics]
        nodes = np.array([data_used, costs]).transpose()
        self.curve.setData(nodes)
        self.lr.setBounds((0, data_used[-1]))
