import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import PlotWidget

import ai
from ui.metrics_dispatcher import Hub


class DistributionHub(Hub):
    def __init__(self) -> None:
        super().__init__()
        self._neurons = []
        self._outputs = []

    def update_data(self, metrics: list[ai.TrainMetric]):
        for m in metrics:
            neurons = np.concatenate([np.indices(o.shape)[0] for o in m.outputs])
            outputs = np.concatenate([o for o in m.outputs])
            self._neurons.append(neurons)
            self._outputs.append(outputs)

    def calc(self, left, right, resolution):
        neurons = np.concatenate(self._neurons[left:right])
        outputs = np.concatenate(self._outputs[left:right])
        bin_x = resolution
        bin_y = np.linspace(0, 1, resolution)
        hist, _, _ = np.histogram2d(neurons, outputs, bins=(bin_x, bin_y), normed=False)
        return hist


class DistributionWidget(QWidget):
    def __init__(self, hub: DistributionHub, resolution=100) -> None:
        super().__init__()
        self._hub = hub
        self._resolution = resolution

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        plot = PlotWidget()
        self._item = plot.getPlotItem()
        plot.setTitle("Activation by neuron")
        plot.setLabel('left', "Activation")
        plot.setLabel('bottom', "Neuron")

        self._bar = pg.ColorBarItem(values=(0, 10))
        self._img = pg.ImageItem()
        self._image_transform = QTransform()
        self._image_transform.scale(1 / self._resolution, 1 / self._resolution)
        self._img.setTransform(self._image_transform)
        self._item.addItem(self._img)

        layout.addWidget(plot)
        self.setLayout(layout)

    def update_data(self, left, right):
        if left is not None and right is not None and self._resolution:
            hist = self._hub.calc(left, right, self._resolution)
            self._img.setImage(image=hist)
        else:
            self._img.clear()
        self._bar.setImageItem(self._img, insert_in=self._item)
