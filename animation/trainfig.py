from queue import Empty

import matplotlib.animation as ani
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl

from animation.cost_comp_sa import CostCompSA
from animation.cost_sa import CostSA
from animation.distribution_sa import DistributionSA
from animation.y_avg_sa import YAvgSA


class TrainFig:
    def __init__(self, item_store, animation_interval, metrics_range=1):
        super().__init__()
        self.fig = pl.figure()
        gs = gridspec.GridSpec(2, 3)
        cost_ax = pl.subplot(gs[0, :])
        cost_comp_ax = pl.subplot(gs[1, 0])
        correlation_ax = pl.subplot(gs[1, 1])
        distribution_ax = pl.subplot(gs[1, 2])
        self.animation = ani.FuncAnimation(self.fig, self._animate, interval=animation_interval)
        self._queue = item_store
        self._retained_metrics = []
        self._sub_animations = [CostSA(cost_ax, metrics_range=metrics_range),
                                CostCompSA(cost_comp_ax, metrics_range=metrics_range),
                                YAvgSA(correlation_ax, metrics_range=metrics_range),
                                DistributionSA(distribution_ax)]

    def _animate(self, _):
        try:
            print('FETCHING METRICS')
            metrics = self._queue.get_nowait()
        except Empty:
            return
        for a in self._sub_animations:
            a.tick(metrics)
