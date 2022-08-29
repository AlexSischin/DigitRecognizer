import statistics

import numpy as np

from ai import TrainMetric, square_mean_costs


def get_avg_components(list_of_vectors):
    return [statistics.mean(e) for e in zip(*list_of_vectors)]


class CostCompSA:
    def __init__(self, ax, metrics_range=1):
        super().__init__()
        self._ax = ax
        self._metrics_range = metrics_range
        self._metrics_buff: list[TrainMetric] = []
        self._cost_components_avg = None
        self._neuron_count = 0

    def tick(self, metrics: list[TrainMetric]):
        self._update_state(metrics)
        self._draw()

    def _update_state(self, metrics: list[TrainMetric]):
        self._metrics_buff.extend(metrics)
        extra_metrics_count = len(self._metrics_buff) - self._metrics_range
        if extra_metrics_count > 0:
            self._metrics_buff = self._metrics_buff[extra_metrics_count::]

        all_cost_components = [square_mean_costs(m.costs) for m in self._metrics_buff]
        self._cost_components_avg = np.mean(np.array(all_cost_components), axis=0)
        self._neuron_count = len(self._cost_components_avg)

    def _draw(self):
        self._ax.cla()
        self._ax.grid(axis='y')
        self._ax.set_title('Average of cost components')
        self._ax.set_xlabel('Neuron')
        self._ax.set_ylabel('Average cost')
        indexes = np.arange(self._neuron_count)
        width = 0.5
        self._ax.bar(indexes, self._cost_components_avg, color='#f95d6a', width=width)
