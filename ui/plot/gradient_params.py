from enum import Enum

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QComboBox, QSizePolicy

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


class GradientParams(QWidget):
    sigOptionsUpdated = pyqtSignal(Component, Mode, Aggregation, int)
    sigOptionsDrop = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._component = Component.WEIGHTS
        self._mode = Mode.GRADIENT
        self._aggregation = Aggregation.NONE
        self._layer = None
        self._layer_count = 0

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel('Component: '), 0, 0)
        layout.addWidget(self._init_component_cb(), 0, 1)

        layout.addWidget(QLabel('Mode: '), 1, 0)
        layout.addWidget(self._init_mode_cb(), 1, 1)

        layout.addWidget(QLabel('Aggregation: '), 2, 0)
        layout.addWidget(self._init_aggregation_cb(), 2, 1)

        layout.addWidget(QLabel('Layer: '), 3, 0)
        layout.addWidget(self._init_layer_cb(), 3, 1)

        self.setLayout(layout)

    def _init_component_cb(self):
        cb = QComboBox()
        cb.addItem(Component.WEIGHTS.value)
        cb.addItem(Component.BIASES.value)
        cb.currentTextChanged.connect(self._set_component_str)
        self._component_cb = cb
        return self._component_cb

    def _set_component_str(self, component_str: str):
        component = find_enum_by_value(Component, component_str)
        self._set_component(component)

    def _set_component(self, component: Component):
        if not isinstance(component, Component):
            raise ValueError(f'Component must be instance of Component. Got: {component}')
        self._component = component
        self._emit_options_changed_or_dropped()

    def _init_mode_cb(self):
        cb = QComboBox()
        cb.addItem(Mode.GRADIENT.value)
        cb.addItem(Mode.STATE.value)
        cb.currentTextChanged.connect(self._set_mode_str)
        self._mode_cb = cb
        return self._mode_cb

    def _set_mode_str(self, mode_str: str):
        mode = find_enum_by_value(Mode, mode_str)
        self._set_mode(mode)

    def _set_mode(self, mode: Mode):
        if not isinstance(mode, Mode):
            raise ValueError(f'Mode must be instance of Mode. Got: {mode}')
        self._mode = mode
        self._emit_options_changed_or_dropped()

    def _emit_options_changed_or_dropped(self):
        params = (self._component, self._mode, self._aggregation, self._layer)
        if None in params:
            self.sigOptionsDrop.emit()
        else:
            self.sigOptionsUpdated.emit(*params)

    def _init_aggregation_cb(self):
        cb = QComboBox()
        cb.addItem(Aggregation.NONE.value)
        cb.addItem(Aggregation.SUM.value)
        cb.addItem(Aggregation.MEAN.value)
        cb.currentTextChanged.connect(self._set_aggregation_str)
        self._aggregation_cb = cb
        return self._aggregation_cb

    def _set_aggregation_str(self, aggregation_str: str):
        aggregation = find_enum_by_value(Aggregation, aggregation_str)
        self._set_aggregation(aggregation)

    def _set_aggregation(self, aggregation: Aggregation):
        if not isinstance(aggregation, Aggregation):
            raise ValueError(f'Aggregation must be instance of Aggregation. Got: {aggregation}')
        self._aggregation = aggregation
        self._emit_options_changed_or_dropped()

    def _init_layer_cb(self):
        cb = QComboBox()
        cb.currentTextChanged.connect(self._set_layer_str)
        self._layer_cb = cb
        return self._layer_cb

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
        self._emit_options_changed_or_dropped()

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
