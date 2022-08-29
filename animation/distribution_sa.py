import numpy as np
from matplotlib import colors as col, pyplot as plt, colorbar as cob

from ai import TrainMetric


def _get_colorbar(cmap, ax):
    cmap = plt.get_cmap(cmap, 100_000)
    norm = col.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, ax=ax)
    cb.set_label('Entries')
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['0%', '50%', '100%'])
    return cb


class DistributionSA:
    def __init__(self, ax, resolution=10, cmap='magma'):
        super().__init__()
        self._ax = ax
        self._resolution = resolution
        self._cmap = cmap
        self._cbar: cob.Colorbar = _get_colorbar(cmap, ax)

    def tick(self, metrics: list[TrainMetric]):
        self._update_state(metrics)
        self._draw()

    def _update_state(self, metrics: list[TrainMetric]):
        neurons = []
        activations = []
        self._neuron_count = len(metrics[0].outputs[0])
        for metric in metrics:
            for output in metric.outputs:
                for neuron, activation in enumerate(output):
                    neurons.append(neuron)
                    activations.append(activation)
        self._neurons = np.array(neurons)
        self._activations = np.array(activations)

    def _draw(self):
        self._ax.set_title('Activation distribution')
        self._ax.set_xlabel('Neuron')
        self._ax.set_ylabel('Activation')
        self._ax.hist2d(self._neurons - 0.5, self._activations, bins=(self._neuron_count, self._resolution),
                        range=((-0.5, self._neuron_count - 0.5), (0, 1)), cmap=self._cmap)
