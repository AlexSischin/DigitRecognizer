from ai import TrainMetric, square_mean_costs


class CostSA:
    def __init__(self, ax, metrics_range=None):
        super().__init__()
        self._ax = ax
        self._next_data_used = 0
        self._data_used = []
        self._inverted_costs = []
        self._metrics_range = metrics_range

    def tick(self, metrics: list[TrainMetric]):
        self._update_state(metrics)
        self._draw()

    def _update_state(self, metrics: list[TrainMetric]):
        for metric in metrics:
            inverted_cost = 1 / sum(square_mean_costs(metric.costs))
            self._data_used.append(self._next_data_used)
            self._next_data_used += len(metric.inputs)
            self._inverted_costs.append(inverted_cost)

    def _draw(self):
        self._ax.cla()
        self._ax.grid(axis='y')
        self._ax.set_title('Inverted cost by training examples')
        self._ax.set_xlabel('Training examples')
        self._ax.set_ylabel('Inverted cost')
        if self._metrics_range:
            left_metric_ind = max(0, len(self._data_used) - self._metrics_range - 1)
            x_left = self._data_used[left_metric_ind]
            x_right = self._data_used[-1]
            self._ax.axvspan(x_left, x_right, facecolor='#004c6d', alpha=0.2, zorder=-100, label='Sample range')
            self._ax.legend()
        self._ax.plot(self._data_used, self._inverted_costs, color='#004c6d', linewidth=1)
