import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import PlotWidget

import ai
from ui.metrics_dispatcher import Hub


class RecentCostHub(Hub):
    def __init__(self) -> None:
        super().__init__()
        self._costs = []

    def update_data(self, metrics: list[ai.TrainMetric]):
        self._costs.extend([(m.data_used, m.cost) for m in metrics])

    def calc(self, max_range):
        left = max(0, len(self._costs) - max_range)
        right = len(self._costs)

        costs, avg_costs = [], []
        if right > left:
            costs = np.array(self._costs[left:right])

            if right > left + 1:
                avg_cost = np.mean(costs[:, 1])
                data_used_left, data_used_right = costs[0, 0], costs[-1, 0]
                avg_costs = np.array([[data_used_left, avg_cost], [data_used_right, avg_cost]])
        return costs, avg_costs


class RecentCostWidget(QWidget):
    def __init__(self, hub: RecentCostHub, max_range=20) -> None:
        super().__init__()

        self._hub = hub
        self._max_range = max_range

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        plot = PlotWidget()
        plot.addLegend()
        plot.setTitle("Recent cost by train data")
        plot.setLabel('left', "Cost")
        plot.setLabel('right', None)
        plot.setLabel('bottom', "Train data")

        self._curve = plot.plot(pen=(0, 128, 0), symbol='t', symbolBrush=(0, 128, 0))
        self._avg_curve = plot.plot(name='Average', pen=(119, 172, 48))
        plot.showGrid(x=True, y=True)

        layout.addWidget(plot)
        self.setLayout(layout)

    def update_data(self):
        if self._max_range is not None:
            nodes, avg_nodes = self._hub.calc(self._max_range)
        else:
            nodes, avg_nodes = [], []
        self._curve.setData(nodes)
        self._avg_curve.setData(avg_nodes)
