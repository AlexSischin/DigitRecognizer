import pyqtgraph as pg

import ai
from ui.plot.base_plot import BasePlot


class WGradPlot(BasePlot):
    def __init__(self):
        super().__init__()

        self._layer = 0
        self._metric_l = None
        self._metric_r = None

        self.setStyleSheet("border: 1px solid black;")
        self.setTitle("Weight gradient")
        self.setLabel('left', "Left layer")
        self.setLabel('bottom', "Right layer")

        cm = pg.colormap.get('CET-D4')
        self._bar = pg.ColorBarItem(values=(-10**-1, 10**-1), rounding=10**-9, colorMap=cm)
        self._img = pg.ImageItem()
        self._item.addItem(self._img)

    def set_data(self, metrics: list[ai.TrainMetric]):
        if len(metrics):
            self._metric_l = metrics[0]
            self._metric_r = metrics[-1]
        else:
            self._metric_l = None
            self._metric_r = None
        self.update_image()

    def update_image(self):
        sum_grad = None
        if self._metric_l and self._metric_r and self._layer >= 0:
            l_grad = self._metric_l.w_gradient[self._layer]
            l_state = self._metric_l.w[self._layer]
            r_state = self._metric_r.w[self._layer]
            sum_grad = l_grad + l_state - r_state

        if sum_grad is not None:
            self._img.setImage(image=sum_grad)
        else:
            self._img.clear()
        self._bar.setImageItem(self._img, insert_in=self._item)

    def set_layer(self, layer: int):
        self._layer = layer - 1
        self.update_image()
