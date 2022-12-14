from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMdiArea, QSplitter, QMdiSubWindow

import ai
from ui.ai_hub import AiHub
from ui.metrics_dispatcher import MetricsDispatchWorkerThread
from ui.plot.correlation import CorrelationWidget, CorrelationHub
from ui.plot.cost import CostWidget, CostHub
from ui.plot.distribution import DistributionWidget, DistributionHub
from ui.plot.grad_len import GradLenWidget, GradLenHub
from ui.plot.gradient import GradientHub, GradientWidget
from ui.plot.recent_cost import RecentCostWidget, RecentCostHub
from ui.version_hub import VersionHub, Approx


class CentralWidget(QWidget):
    sigAiVersionSelected = pyqtSignal(float)
    sigTrainRun = pyqtSignal()
    sigTrainFinished = pyqtSignal()
    sigRefreshed = pyqtSignal()
    sigMetricsUpdated = pyqtSignal()
    sigRegionUpdated = pyqtSignal(int, int)

    def __init__(self, queue, layer_count):
        super().__init__()
        self._queue = queue
        self._layer_count = layer_count
        self._left = None
        self._right = None

        self._init_hubs()
        self._init_threads()

        self.setLayout(self._init_layout())

        self.add_recent_cost_widget()
        self.tile()

    def _init_hubs(self):
        # Util
        self._version_hub = VersionHub()
        self._ai_hub = AiHub()

        # Sub widgets
        self._cost_hub = CostHub()
        self._recent_cost_hub = RecentCostHub()
        self._correlation_hub = CorrelationHub()
        self._distribution_hub = DistributionHub()
        self._grad_len_hub = GradLenHub()
        self._gradient_hub = GradientHub()

    def _init_threads(self):
        # Metrics dispatch
        self._metrics_dispatcher = MetricsDispatchWorkerThread(self._queue, hubs=(
            self._version_hub, self._ai_hub, self._cost_hub, self._recent_cost_hub, self._correlation_hub,
            self._distribution_hub, self._grad_len_hub, self._gradient_hub
        ))
        self._metrics_dispatcher.started.connect(self.sigTrainRun.emit)
        self._metrics_dispatcher.updated.connect(self.sigMetricsUpdated.emit)
        self._metrics_dispatcher.finished.connect(self.sigTrainFinished.emit)
        self._metrics_dispatcher.finished.connect(self._metrics_dispatcher.deleteLater)
        self._metrics_dispatcher.start()

    def _init_layout(self):
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Vertical)
        splitter.setOpaqueResize(False)
        splitter.addWidget(self._init_dispatcher_widget())
        splitter.addWidget(self._init_mdi_widget())
        splitter.setSizes((200, 200))
        layout.addWidget(splitter)
        return layout

    def _init_dispatcher_widget(self):
        self._dispatcher_widget = CostWidget(self._cost_hub)
        self.sigRefreshed.connect(self._dispatcher_widget.refresh)
        self._dispatcher_widget.sigRegionChanged.connect(self._update_region)
        self._dispatcher_widget.sigSpotSelected.connect(self.sigAiVersionSelected.emit)
        return self._dispatcher_widget

    def _init_mdi_widget(self):
        self._mdi = QMdiArea()
        self._mdi.setMinimumHeight(200)
        return self._mdi

    def add_recent_cost_widget(self):
        widget = RecentCostWidget(self._recent_cost_hub)
        widget.update_data()
        self.sigMetricsUpdated.connect(widget.update_data)
        self._add_mdi_widget(widget)

    def add_correlation_widget(self):
        widget = CorrelationWidget(self._correlation_hub)
        widget.update_data(self._left, self._right)
        self.sigRegionUpdated.connect(widget.update_data)
        self._add_mdi_widget(widget)

    def add_distribution_widget(self):
        widget = DistributionWidget(self._distribution_hub)
        widget.update_data(self._left, self._right)
        self.sigRegionUpdated.connect(widget.update_data)
        self._add_mdi_widget(widget)

    def add_grad_len_widget(self):
        widget = GradLenWidget(self._grad_len_hub)
        widget.update_data(self._left, self._right)
        self.sigRegionUpdated.connect(widget.update_data)
        self._add_mdi_widget(widget)

    def add_gradient_widget(self):
        widget = GradientWidget(self._gradient_hub)
        widget.set_layer_count(self._layer_count)
        widget.set_layer(self._layer_count - 1)
        widget.set_region(self._left, self._right)
        self.sigRegionUpdated.connect(widget.set_region)
        self._add_mdi_widget(widget)

    def _add_mdi_widget(self, widget):
        sub = QMdiSubWindow()
        sub.setWidget(widget)
        sub.setAttribute(Qt.WA_DeleteOnClose)
        self._mdi.addSubWindow(sub)
        sub.show()

    def tile(self):
        self._mdi.tileSubWindows()

    def _update_region(self, left_du, right_du):
        left = self._version_hub.get_version(left_du, approx=Approx.GE)
        right = self._version_hub.get_version(right_du, approx=Approx.LE) + 1

        if left is None or right is None:
            left, right = None, None

        if self._left != left or self._right != right:
            self._left, self._right = left, right
            self.sigRegionUpdated.emit(self._left, self._right)

    def get_ai_v_by_duv(self, data_used_version: int, act_funcs: tuple[ai.ActivationFunction]) -> ai.Ai:
        v = self._version_hub.get_version(data_used_version)
        w, b = self._ai_hub.get_version(v)
        return ai.Ai(weights=w, biases=b, activation_functions=act_funcs)
