import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QGridLayout, QVBoxLayout
from pyqtgraph import PlotWidget

from distribution_viewer.functions import Function
from distributions import LayerDistributions
from utils.zip_utils import zip3


def wrap_widget(widget):
    wrapper = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)
    wrapper.setLayout(layout)
    return wrapper


class LayerDistributionWidget(QWidget):
    def __init__(self, layer: int, distributions: LayerDistributions, domain: np.ndarray) -> None:
        super().__init__()
        self._layer = layer
        self._distributions = distributions
        self._domain = domain

        a_prev_plot = self._create_plot(self._distributions.A_prev, f'A_{layer - 1}')
        w_plot = self._create_plot(self._distributions.W, f'W_{layer}')
        b_plot = self._create_plot(self._distributions.B, f'B_{layer}')

        p_plot = self._create_plot(self._distributions.P, f'P_{layer} = A_{layer - 1} * W_{layer}')
        s_plot = self._create_plot(self._distributions.S, f'S_{layer} = Sum(P_{layer})')

        z_plot = self._create_plot(self._distributions.Z, f'Z_{layer} = S_{layer} + B_{layer}')
        a_plot = self._create_plot(self._distributions.A, f'A_{layer} = F_{layer}(Z_{layer})')

        self._layout = QGridLayout()

        self._layout.addWidget(wrap_widget(a_prev_plot), 0, 0, 1, 2)
        self._layout.addWidget(wrap_widget(w_plot), 0, 2, 1, 2)
        self._layout.addWidget(wrap_widget(b_plot), 0, 4, 1, 2)

        self._layout.addWidget(wrap_widget(p_plot), 1, 0, 1, 3)
        self._layout.addWidget(wrap_widget(s_plot), 1, 3, 1, 3)

        self._layout.addWidget(wrap_widget(z_plot), 2, 0, 1, 3)
        self._layout.addWidget(wrap_widget(a_plot), 2, 3, 1, 3)

        self.setLayout(self._layout)

    def _create_plot(self, func: Function, label):
        x = self._domain
        y = func.at_vec(x)
        plot = PlotWidget()
        plot.setTitle(label)
        plot.plot(x=x, y=y)
        plot.showGrid(x=True, y=True)
        return plot


class DistributionViewerWidget(QTabWidget):
    def __init__(self,
                 layers: tuple[int, ...],
                 distributions: tuple[LayerDistributions, ...],
                 domain: np.ndarray) -> None:
        super().__init__()
        self._init_tabs(layers, distributions, domain)

    def _init_tabs(self, layers, distributions, domain):
        for i_prev, (l_prev, l, distribution) in enumerate(zip3(layers[:-1], layers[1:], distributions)):
            i = i_prev + 1
            label = f'Layer {i}[{l_prev}-{l}]'
            distribution_widget = LayerDistributionWidget(i, distribution, domain)
            self.addTab(distribution_widget, label)


class DistributionViewerApp(QApplication):
    def __init__(self,
                 layers: tuple[int, ...],
                 distributions: tuple[LayerDistributions, ...],
                 domain: np.ndarray) -> None:
        super().__init__([])
        self._window = QMainWindow()
        self._window.setWindowTitle('Initial distributions')
        widget = DistributionViewerWidget(layers, distributions, domain)
        self._window.setCentralWidget(widget)

    def exec(self) -> int:
        self._window.show()
        return super().exec()
