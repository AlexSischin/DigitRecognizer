from numbers import Number
from typing import Iterable

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSizePolicy, QFrame
from pyqtgraph import PlotWidget

import ai
from ui.metrics_dispatcher import Hub
from ui.plot.gradient_info import GradientInfo
from ui.plot.gradient_params import Component, Mode, Aggregation, GradientParams


def flatten(it: Iterable | np.ndarray) -> np.ndarray:
    if isinstance(it, np.ndarray):
        return it.flatten()
    elif isinstance(it, Iterable):
        result = np.array([])
        for i in it:
            if isinstance(i, Number):
                result = np.append(result, i)
            elif isinstance(i, Iterable):
                result = np.concatenate((result, flatten(i)))
            else:
                raise ValueError('Value must contain only numbers and iterables at any depth')
        return result
    else:
        raise ValueError('Value must be iterable')


def get_distribution_params(a: np.ndarray):
    return np.array([a.size, np.mean(a), np.std(a)])


def get_distribution_params_for_batch(*arrays):
    return [get_distribution_params(a) for a in arrays]


def add_dimension_for_batch(*arrays, axis=0):
    return [np.expand_dims(a, axis=axis) for a in arrays]


def combine_distributions_params(
        sizes: np.ndarray,
        means: np.ndarray,
        sds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sizes.ndim != 1 or means.ndim != 1 or sds.ndim != 1:
        raise ValueError('All arrays must be 1d')
    if sizes.size != means.size or sizes.size != sds.size:
        raise ValueError('All arrays must be the same size')

    sizes_c = np.sum(sizes)
    means_c = np.sum(sizes * means) / sizes_c
    sds_c = (np.sum((means ** 2 + sds ** 2) * sizes) / sizes_c - means_c ** 2) ** 0.5
    return sizes_c, means_c, sds_c


def get_info_aggr(data_slice: np.ndarray, aggregation: Aggregation):
    if aggregation is Aggregation.NONE:
        return data_slice
    elif aggregation is Aggregation.SUM:
        return np.sum(data_slice, axis=0)
    elif aggregation is Aggregation.MEAN:
        return np.mean(data_slice, axis=0)
    else:
        raise ValueError(f'Invalid aggregation: {aggregation}')


class GradientHub(Hub):

    def __init__(self) -> None:
        self._data_used = []  # size=time
        self._ws = []  # size=(time, layer, (y, x));  y and x are different for different layers
        self._wg = []  # size=(time, layer, (y, x));  y and x are different for different layers
        self._bs = []  # size=(time, layer, (y, x));  y and x are different for different layers
        self._bg = []  # size=(time, layer, (y, x));  y and x are different for different layers
        self._ws_stats = []  # size=(time, layer, stat); stat=3
        self._wg_stats = []  # size=(time, layer, stat); stat=3
        self._bs_stats = []  # size=(time, layer, stat); stat=3
        self._bg_stats = []  # size=(time, layer, stat); stat=3

    def update_data(self, metrics: list[ai.TrainMetric]):
        for m in metrics:
            self._data_used.append(m.data_used)
            self._ws.append(m.w)
            self._wg.append(m.w_gradient)

            ws_stats = get_distribution_params_for_batch(*m.w)
            self._ws_stats.append(ws_stats)

            wg_stats = get_distribution_params_for_batch(*m.w_gradient)
            self._wg_stats.append(wg_stats)

            bs_stats = get_distribution_params_for_batch(*m.b)
            self._bs_stats.append(bs_stats)

            bg_stats = get_distribution_params_for_batch(*m.b_gradient)
            self._bg_stats.append(bg_stats)

            bs_2d = add_dimension_for_batch(*m.b)
            self._bs.append(bs_2d)

            bg_2d = add_dimension_for_batch(*m.b_gradient)
            self._bg.append(bg_2d)

    def get_info(self,
                 left: int,
                 right: int,
                 layer: int,
                 component: Component,
                 mode: Mode,
                 aggregation: Aggregation
                 ) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
        data_source = {
            (Component.WEIGHTS, Mode.GRADIENT): (self._wg, self._wg_stats),
            (Component.WEIGHTS, Mode.STATE): (self._ws, self._ws_stats),
            (Component.BIASES, Mode.GRADIENT): (self._bg, self._bg_stats),
            (Component.BIASES, Mode.STATE): (self._bs, self._bs_stats),
        }
        # size=(time, layer, (y, x));  y and x are different for different layers
        data, data_stats = data_source[component, mode]
        data_slice, data_stats_slice = data[left:right], data_stats[left:right]

        # size=(time, y, x)
        data_layer_slice = np.array([d[layer] for d in data_slice])

        # size=(y, x) or (time, y, x) no aggregation
        aggregated_data = get_info_aggr(data_layer_slice, aggregation)

        # shape=(layer, stat, time); stat=3
        grouped_stats = np.array(data_stats_slice).transpose(1, 2, 0)

        # shape=(layer, stat); stat=3
        combined_stats = [combine_distributions_params(s[0], s[1], s[2]) for s in grouped_stats]

        # shape=stat; stat=3
        layer_combined_stats = combined_stats[layer]

        # shape=(stat, layer); stat=3
        grouped_combined_stats = np.array(combined_stats).transpose(1, 0)

        # shape=stat; stat=3
        combined_combined_stats = combine_distributions_params(
            grouped_combined_stats[0],
            grouped_combined_stats[1],
            grouped_combined_stats[2]
        )

        return aggregated_data, layer_combined_stats, combined_combined_stats

    def get_x_vals(self, left, right):
        return np.array(self._data_used[left:right])


class GradientWidget(QWidget):
    def __init__(self, hub: GradientHub):
        super().__init__()

        self._hub = hub

        self._left = None
        self._right = None
        self._component = None
        self._mode = None
        self._aggregation = None
        self._layer = 0
        self._layer_count = 0

        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 0, 0)
        layout.addWidget(self._init_left_bar_widget(), alignment=Qt.AlignTop)
        layout.addWidget(self._init_img_view_widget())
        self.setLayout(layout)

    def _init_left_bar_widget(self):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        layout = QVBoxLayout()

        self._gradient_params = GradientParams()
        self._gradient_params.setContentsMargins(0, 0, 0, 0)
        self._gradient_params.sigOptionsUpdated.connect(self._update_params)
        self._gradient_params.sigOptionsDrop.connect(self._drop_params)
        layout.addWidget(self._gradient_params)

        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(h_line)

        self._gradient_info = GradientInfo()
        self._gradient_info.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._gradient_info)

        widget.setLayout(layout)
        return widget

    def _init_img_view_widget(self):
        self._imv = pg.ImageView()
        self._imv.getView().setAspectLocked(False)

        plot = PlotWidget()
        self._item = plot.getPlotItem()
        plot.setTitle("Weight gradient")
        plot.setLabel('left', "Left layer")
        plot.setLabel('bottom', "Right layer")

        self._imv.setPredefinedGradient('bipolar')

        return self._imv

    def _update_data(self):
        if self._left is not None \
                and self._right is not None \
                and self._left != self._right \
                and self._layer is not None \
                and self._component is not None \
                and self._mode is not None \
                and self._aggregation is not None:
            info, layer_stats, stats = self._hub.get_info(
                self._left, self._right, self._layer, self._component, self._mode, self._aggregation
            )
            x_vals = self._hub.get_x_vals(self._left, self._right)
            self._imv.setImage(info, xvals=x_vals)
            self._gradient_info.set_layer_info(*layer_stats)
            self._gradient_info.set_info(*stats)
        else:
            self._imv.clear()
            self._gradient_info.clear_layer_info()
            self._gradient_info.clear_info()

    def set_region(self, left: int, right: int):
        self._left, self._right = left, right
        self._update_data()

    def _update_params(self, component: Component, mode: Mode, aggregation: Aggregation, layer: int | None):
        self._component = component
        self._mode = mode
        self._aggregation = aggregation
        self._layer = layer
        self._update_data()

    def _drop_params(self):
        self._component = self._mode = self._aggregation = self._layer = None
        self._update_data()

    def set_layer_count(self, layer_count: int):
        self._gradient_params.set_layer_count(layer_count)

    def set_layer(self, layer: int | None):
        self._gradient_params.set_layer(layer)
