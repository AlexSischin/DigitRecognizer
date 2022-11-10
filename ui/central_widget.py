import math

import pyqtgraph as pg

import ai
from ui.plot.correlation import CorrelationPlot
from ui.plot.cost import CostPlot
from ui.plot.distribution import DistributionPlot
from ui.plot.recent_cost import RecentCostPlot


class CentralWidget(pg.GraphicsLayoutWidget):

    def __init__(self, metrics_buff: list[ai.TrainMetric], train_data_chunk_size: int, parent=None):
        super().__init__(parent)
        self._metrics_buff = metrics_buff
        self._train_data_chunk_size = train_data_chunk_size
        self._last_region = None

        self._init_plots()

    def _init_plots(self):
        # All costs
        cost_plot = self.addPlot(colspan=2)
        self._cost_plot = CostPlot(cost_plot, self._train_data_chunk_size)
        self._cost_plot.lr.sigRegionChanged.connect(self.update_region)

        # Recent costs
        recent_cost_plot = self.addPlot()
        self._recent_cost_plot = RecentCostPlot(recent_cost_plot)

        # ------------------
        self.nextRow()

        # Correlation
        correlation_plot = self.addPlot()
        self._correlation_plot = CorrelationPlot(correlation_plot)

        # Distribution
        distribution_plot = self.addPlot()
        self._distribution_plot = DistributionPlot(distribution_plot)

    def update_region(self):
        left_data_bound, right_data_bound = self._cost_plot.lr.getRegion()
        left_metrics_bound = max(0,
                                 math.ceil(left_data_bound / self._train_data_chunk_size))
        right_metrics_bound = min(len(self._metrics_buff),
                                  math.floor(right_data_bound / self._train_data_chunk_size) + 1)
        region = (left_metrics_bound, right_metrics_bound)
        if region != self._last_region:
            self._last_region = region
            region_metrics = self._metrics_buff[left_metrics_bound:right_metrics_bound]
            self._correlation_plot.update(region_metrics)
            self._distribution_plot.update(region_metrics)

    def update_cost_plot(self):
        data_used = [i * self._train_data_chunk_size for i, _ in enumerate(self._metrics_buff)]
        self._cost_plot.update(data_used, self._metrics_buff)

    def update_metrics(self):
        data_used = [i * self._train_data_chunk_size for i, _ in enumerate(self._metrics_buff)][-50:]
        recent_metrics = self._metrics_buff[-50:]
        self._recent_cost_plot.update(data_used, recent_metrics)
