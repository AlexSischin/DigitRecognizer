import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QTransform

import ai
from ui.plot.base_plot import BasePlot


class DistributionPlot(BasePlot):
    def __init__(self, resolution=100) -> None:
        super().__init__()
        self._resolution = resolution

        self.setStyleSheet("border: 1px solid black;")
        self.setTitle("Activation by neuron")
        self.setLabel('left', "Activation")
        self.setLabel('bottom', "Neuron")

        self._bar = pg.ColorBarItem(values=(0, 10))
        self._img = pg.ImageItem()
        self.image_transform = QTransform()
        self.image_transform.scale(1/self._resolution, 1/self._resolution)
        self._img.setTransform(self.image_transform)
        self._item.addItem(self._img)

    def set_data(self, metrics: list[ai.TrainMetric]):
        neurons = []
        outputs = []
        for metric in metrics:
            for output in metric.outputs:
                for neuron, activation in enumerate(output):
                    neurons.append(neuron)
                    outputs.append(activation)
        bin_x = self._resolution
        bin_y = np.linspace(0, 1, self._resolution)
        hist, _, _ = np.histogram2d(neurons, outputs, bins=(bin_x, bin_y), normed=False)
        self._img.setImage(image=hist)
        self._bar.setImageItem(self._img, insert_in=self._item)
