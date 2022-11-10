import numpy as np

import ai


def square_mean_costs(costs: list[np.ndarray]) -> np.ndarray:
    costs_array = np.array(costs)
    squared_costs = np.square(costs_array)
    return np.mean(squared_costs, axis=0)


class RecentCostPlot:
    def __init__(self, plot) -> None:
        super().__init__()
        self._plot = plot

        self._plot.addLegend()
        self._plot.setTitle("Recent cost by train data")
        self._plot.setLabel('left', "Cost")
        self._plot.setLabel('right', None)
        self._plot.setLabel('bottom', "Train data")
        self._curve = self._plot.plot(pen=(0, 128, 0), symbol='t', symbolBrush=(0, 128, 0))
        self._avg_curve = self._plot.plot(name='Average', pen=(119, 172, 48))
        self._plot.showGrid(x=True, y=True)

    def update(self, data_used: list[int], last_metrics: list[ai.TrainMetric]):
        costs = np.array([sum(square_mean_costs(m.costs)) for m in last_metrics])
        nodes = np.array([data_used, costs]).transpose()
        self._curve.setData(nodes)
        if last_metrics:
            avg_cost = np.mean(costs)
            nodes = np.array([[data_used[0], avg_cost], [data_used[-1], avg_cost]])
            self._avg_curve.setData(nodes)

