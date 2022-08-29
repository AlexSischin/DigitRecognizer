import statistics

import numpy as np

from ai import TrainMetric


def get_avg_components(list_of_vectors):
    return [statistics.mean(e) for e in zip(*list_of_vectors)]


class YAvgSA:
    def __init__(self, ax, metrics_range=1):
        super().__init__()
        self._ax = ax
        self._metrics_range = metrics_range
        self._metrics_buff: list[TrainMetric] = []
        self._y_act_avg = None
        self._y_exp_avg = None
        self._neuron_count = 0

    def tick(self, metrics: list[TrainMetric]):
        self._update_state(metrics)
        self._draw()

    def _update_state(self, metrics: list[TrainMetric]):
        self._metrics_buff.extend(metrics)
        extra_metrics_count = len(self._metrics_buff) - self._metrics_range
        if extra_metrics_count > 0:
            self._metrics_buff = self._metrics_buff[extra_metrics_count::]

        all_y_act = [o for m in self._metrics_buff for o in m.outputs]
        all_y_exp = [e for m in self._metrics_buff for e in m.expected]
        self._y_act_avg = np.mean(np.array(all_y_act), axis=0)
        self._y_exp_avg = np.mean(np.array(all_y_exp), axis=0)
        self._neuron_count = len(self._y_act_avg)

    def _draw(self):
        self._ax.cla()
        self._ax.grid(axis='y')
        self._ax.set_title('Average outputs')
        self._ax.set_xlabel('Neuron')
        self._ax.set_ylabel('Average activation')
        indexes = np.arange(self._neuron_count)
        width = 0.35
        self._ax.bar(indexes + width / 2, self._y_act_avg, color='#004c6d', width=width, label='Actual')
        self._ax.bar(indexes - width / 2, self._y_exp_avg, color='#ffa600', width=width, label='Expected')
        self._ax.legend()
