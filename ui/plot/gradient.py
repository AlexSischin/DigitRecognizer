from enum import Enum

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QSizePolicy
from pyqtgraph import PlotWidget

import ai
from ui.metrics_dispatcher import Hub
from utils.enum_utils import find_enum_by_value


class Component(Enum):
    WEIGHTS = 'Weights'
    BIASES = 'Biases'


class Mode(Enum):
    GRADIENT = 'Gradient'
    STATE = 'State'


class Aggregation(Enum):
    NONE = 'None'
    SUM = 'Sum'
    MEAN = 'Mean'


def reg_option(d: dict, component: Component, mode: Mode, aggregation: Aggregation):
    def wrapper(fn):
        d[component, mode, aggregation] = fn
        return fn

    return wrapper


class GradientHub(Hub):
    methods = {}

    def __init__(self) -> None:
        self._ws = []
        self._wgs = []
        self._bs = []
        self._bgs = []

    def update_data(self, metrics: list[ai.TrainMetric]):
        self._ws.extend([m.w for m in metrics])
        self._wgs.extend([m.w_gradient for m in metrics])
        self._bs.extend([m.b for m in metrics])
        self._bgs.extend([m.b_gradient for m in metrics])

    @reg_option(methods, Component.WEIGHTS, Mode.GRADIENT, Aggregation.NONE)
    def get_w_grads(self, left, right, layer):
        return np.array([g[layer] for g in self._wgs[left:right]])

    @reg_option(methods, Component.WEIGHTS, Mode.GRADIENT, Aggregation.SUM)
    def get_w_grads_sum(self, left, right, layer):
        grads = self.get_w_grads(left, right, layer)
        return np.sum(grads, axis=0)

    @reg_option(methods, Component.WEIGHTS, Mode.GRADIENT, Aggregation.MEAN)
    def get_w_grads_mean(self, left, right, layer):
        grads = self.get_w_grads(left, right, layer)
        return np.mean(grads, axis=0)

    @reg_option(methods, Component.WEIGHTS, Mode.STATE, Aggregation.NONE)
    def get_w_states(self, left, right, layer):
        return np.array([g[layer] for g in self._ws[left:right]])

    @reg_option(methods, Component.WEIGHTS, Mode.STATE, Aggregation.SUM)
    def get_w_states_sum(self, left, right, layer):
        states = self.get_w_states(left, right, layer)
        return np.sum(states, axis=0)

    @reg_option(methods, Component.WEIGHTS, Mode.STATE, Aggregation.MEAN)
    def get_w_states_mean(self, left, right, layer):
        states = self.get_w_states(left, right, layer)
        return np.mean(states, axis=0)

    @reg_option(methods, Component.BIASES, Mode.GRADIENT, Aggregation.NONE)
    def get_b_grads(self, left, right, layer):
        one_d_grads = np.array([g[layer] for g in self._bgs[left:right]])
        return np.expand_dims(one_d_grads, axis=1)

    @reg_option(methods, Component.BIASES, Mode.GRADIENT, Aggregation.SUM)
    def get_b_grads_sum(self, left, right, layer):
        grads = self.get_b_grads(left, right, layer)
        return np.sum(grads, axis=0)

    @reg_option(methods, Component.BIASES, Mode.GRADIENT, Aggregation.MEAN)
    def get_b_grads_mean(self, left, right, layer):
        grads = self.get_b_grads(left, right, layer)
        return np.mean(grads, axis=0)

    @reg_option(methods, Component.BIASES, Mode.STATE, Aggregation.NONE)
    def get_b_states(self, left, right, layer):
        one_d_states = np.array([g[layer] for g in self._bs[left:right]])
        return np.expand_dims(one_d_states, axis=1)

    @reg_option(methods, Component.BIASES, Mode.STATE, Aggregation.SUM)
    def get_b_states_sum(self, left, right, layer):
        states = self.get_b_states(left, right, layer)
        return np.sum(states, axis=0)

    @reg_option(methods, Component.BIASES, Mode.STATE, Aggregation.MEAN)
    def get_b_states_mean(self, left, right, layer):
        states = self.get_b_states(left, right, layer)
        return np.mean(states, axis=0)


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
        layout.addWidget(self._init_params_widget(), alignment=Qt.AlignTop)
        layout.addWidget(self._init_img_view_widget())
        self.setLayout(layout)

        self._set_component(Component.WEIGHTS)
        self._set_mode(Mode.GRADIENT)
        self._set_aggregation(Aggregation.NONE)

    def _init_params_widget(self):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._init_component_widget())
        layout.addWidget(self._init_mode_widget())
        layout.addWidget(self._init_aggregation_widget())
        layout.addWidget(self._init_layer_widget())
        widget.setLayout(layout)
        return widget

    def _init_component_widget(self):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel('Component:')
        layout.addWidget(label)

        self._component_cb = QComboBox()
        self._component_cb.addItem(Component.WEIGHTS.value)
        self._component_cb.addItem(Component.BIASES.value)
        self._component_cb.currentTextChanged.connect(self._set_component_str)
        layout.addWidget(self._component_cb)

        widget.setLayout(layout)
        return widget

    def _init_mode_widget(self):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel('Mode:')
        layout.addWidget(label)

        self._mode_cb = QComboBox()
        self._mode_cb.addItem(Mode.GRADIENT.value)
        self._mode_cb.addItem(Mode.STATE.value)
        self._mode_cb.currentTextChanged.connect(self._set_mode_str)
        layout.addWidget(self._mode_cb)

        widget.setLayout(layout)
        return widget

    def _init_aggregation_widget(self):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel('Aggregation:')
        layout.addWidget(label)

        self._aggregation_cb = QComboBox()
        self._aggregation_cb.addItem(Aggregation.NONE.value)
        self._aggregation_cb.addItem(Aggregation.MEAN.value)
        self._aggregation_cb.addItem(Aggregation.SUM.value)
        self._aggregation_cb.currentTextChanged.connect(self._set_aggregation_str)
        layout.addWidget(self._aggregation_cb)

        widget.setLayout(layout)
        return widget

    def _init_layer_widget(self):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel('Layer:')
        layout.addWidget(label)

        self._layer_cb = QComboBox()
        self._layer_cb.currentTextChanged.connect(self._set_layer_str)
        layout.addWidget(self._layer_cb)

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
                and self._layer is not None \
                and self._component is not None \
                and self._mode is not None \
                and self._aggregation is not None:
            fn = self._hub.methods[self._component, self._mode, self._aggregation]
            grads = fn(self._hub, self._left, self._right, self._layer)
            self._imv.setImage(grads, xvals=np.linspace(self._left, self._right, grads.shape[0]))
        else:
            self._imv.clear()

    def set_layer_count(self, layer_count: int):
        if layer_count is None or layer_count < 0:
            raise ValueError('Layer count must be a positive integer')
        if self._layer_count != layer_count:
            self._layer_count = layer_count
            self._layer_cb.clear()
            for layer in range(self._layer_count):
                self._layer_cb.addItem(f'{layer}')
            if self._layer is not None and self._layer <= self._layer_count:
                self.set_layer(None)

    def set_region(self, left: int, right: int):
        self._left, self._right = left, right
        self._update_data()

    def set_component(self, component: Component):
        self._set_component(component)

    def _set_component_str(self, component_str: str):
        component = find_enum_by_value(Component, component_str)
        self._set_component(component)

    def _set_component(self, component: Component):
        if not isinstance(component, Component):
            raise ValueError(f'Component must be instance of Component. Got: {component}')
        self._component = component
        self._update_data()

    def set_mode(self, mode: Mode):
        self._set_mode(mode)

    def _set_mode_str(self, mode_str: str):
        mode = find_enum_by_value(Mode, mode_str)
        self._set_mode(mode)

    def _set_mode(self, mode: Mode):
        if not isinstance(mode, Mode):
            raise ValueError(f'Mode must be instance of Mode. Got: {mode}')
        self._mode = mode
        self._update_data()

    def set_aggregation(self, aggregation: Aggregation):
        self._set_aggregation(aggregation)

    def _set_aggregation_str(self, aggregation_str: str):
        aggregation = find_enum_by_value(Aggregation, aggregation_str)
        self._set_aggregation(aggregation)

    def _set_aggregation(self, aggregation: Aggregation):
        if not isinstance(aggregation, Aggregation):
            raise ValueError(f'Aggregation must be instance of Aggregation. Got: {aggregation}')
        self._aggregation = aggregation
        self._update_data()

    def set_layer(self, layer: int | None):
        if not (layer is None or 0 <= layer < self._layer_count):
            raise ValueError(f'Layer must be an integer in range: [0, {self._layer_count}) or None. Got: {layer}')
        self._layer_cb.setCurrentText(str(layer))

    def _set_layer_str(self, layer_str: str | None):
        layer = int(layer_str) if layer_str != '' and layer_str is not None else None
        self._set_layer(layer)

    def _set_layer(self, layer: int | None):
        if not (layer is None or 0 <= layer < self._layer_count):
            raise ValueError(f'Layer must be an integer in range: [0, {self._layer_count}) or None. Got: {layer}')
        if self._layer != layer:
            self._layer = layer
        self._update_data()
