import numpy as np

from ui.metrics_dispatcher import TrainMetric
from ui.plot.base_plot import BasePlot


def square_mean_costs(costs: list[np.ndarray]) -> np.ndarray:
    costs_array = np.array(costs)
    squared_costs = np.square(costs_array)
    return np.mean(squared_costs, axis=0)


class RecentCostPlot(BasePlot):
    def __init__(self) -> None:
        super().__init__()

        self.setStyleSheet("border: 1px solid black;")
        self.addLegend()
        self.setTitle("Recent cost by train data")
        self.setLabel('left', "Cost")
        self.setLabel('right', None)
        self.setLabel('bottom', "Train data")

        self._curve = self.plot(pen=(0, 128, 0), symbol='t', symbolBrush=(0, 128, 0))
        self._avg_curve = self.plot(name='Average', pen=(119, 172, 48))
        self.showGrid(x=True, y=True)

    def set_data(self, last_metrics: list[TrainMetric]):
        nodes = np.array([(m.data_used, sum(square_mean_costs(m.costs))) for m in last_metrics])
        self._curve.setData(nodes)
        if last_metrics:
            avg_cost = np.mean([n[1] for n in nodes])
            avg_nodes = np.array([[last_metrics[0].data_used, avg_cost], [last_metrics[-1].data_used, avg_cost]])
            self._avg_curve.setData(avg_nodes)
