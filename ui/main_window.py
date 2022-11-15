import queue

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import QMainWindow, QToolBar, QAction, QLabel, QStatusBar, QToolButton

import ai
import qrc_resources
from ui.central_widget import CentralWidget
from ui.metrics_dispatcher import MetricsDispatchWorkerThread

# To save from imports optimization by IDEs
qrc_resources = qrc_resources


class MainWindow(QMainWindow):
    _queue:                     queue.Queue
    _metrics_buff:              list[ai.TrainMetric]
    train_data_chunk_size:      int
    _metrics_dispatcher:        MetricsDispatchWorkerThread
    _central_widget:            CentralWidget
    _refresh_action:            QAction
    _main_toolbar:              QToolBar
    _statusbar:                 QStatusBar
    _status_widget:             QLabel

    def __init__(self, metrics_queue, train_data_chunk_size: int, parent=None):
        super().__init__(parent)
        self._queue = metrics_queue
        self._train_data_chunk_size = train_data_chunk_size
        self._metrics_buff = []

        self.setWindowTitle('AI trainer')
        self.resize(1000, 600)

        self._init_threads()
        self._init_widgets()
        self._init_actions()
        self._init_toolbar()
        self._init_statusbar()

    def _init_threads(self):
        # Metrics dispatch
        self._metrics_dispatcher = MetricsDispatchWorkerThread(self._queue, self._metrics_buff)
        self._metrics_dispatcher.started.connect(self.set_train_running_status)
        self._metrics_dispatcher.updated.connect(self.update_recent_costs)
        self._metrics_dispatcher.finished.connect(self.set_train_finished_status)
        self._metrics_dispatcher.finished.connect(self._metrics_dispatcher.deleteLater)
        self._metrics_dispatcher.start()

    def _init_widgets(self):
        # Plots
        self._central_widget = CentralWidget(self._metrics_buff, self._train_data_chunk_size, self)
        self.setCentralWidget(self._central_widget)

    def _init_actions(self):
        # Refresh cost plot
        refresh_icon = QIcon(":refresh-icon")
        refresh_text = '&Refresh costs'
        refresh_tip = 'Refresh costs plot'
        self._refresh_action = QAction(refresh_icon, refresh_text, self)
        self._refresh_action.setShortcuts((QKeySequence.Refresh, 'Ctrl+R'))
        self._refresh_action.setStatusTip(refresh_tip)
        self._refresh_action.setToolTip(refresh_tip)
        self._refresh_action.triggered.connect(self.update_costs)

        # Toggle distribution
        toggle_recent_cost_icon = QIcon(":recent-cost-plot-icon")
        toggle_recent_cost_text = '&Recent cost plot'
        toggle_recent_cost_tip = 'Toggle recent cost plot'
        self._toggle_recent_cost_button = QToolButton(self)
        self._toggle_recent_cost_button.setIcon(toggle_recent_cost_icon)
        self._toggle_recent_cost_button.setText(toggle_recent_cost_text)
        self._toggle_recent_cost_button.setCheckable(True)
        self._toggle_recent_cost_button.setStatusTip(toggle_recent_cost_tip)
        self._toggle_recent_cost_button.setToolTip(toggle_recent_cost_tip)
        self._toggle_recent_cost_button.toggled.connect(self.toggle_recent_cost_plot)
        self._toggle_recent_cost_button.toggle()

        # Toggle correlation
        toggle_correlation_icon = QIcon(":correlation-plot-icon")
        toggle_correlation_text = '&Correlation plot'
        toggle_correlation_tip = 'Toggle correlation plot'
        self._toggle_correlation_button = QToolButton(self)
        self._toggle_correlation_button.setIcon(toggle_correlation_icon)
        self._toggle_correlation_button.setText(toggle_correlation_text)
        self._toggle_correlation_button.setCheckable(True)
        self._toggle_correlation_button.setStatusTip(toggle_correlation_tip)
        self._toggle_correlation_button.setToolTip(toggle_correlation_tip)
        self._toggle_correlation_button.toggled.connect(self.toggle_correlation_plot)
        self._toggle_correlation_button.toggle()

        # Toggle distribution
        toggle_distribution_icon = QIcon(":distribution-plot-icon")
        toggle_distribution_text = '&Distribution plot'
        toggle_distribution_tip = 'Toggle distribution plot'
        self._toggle_distribution_button = QToolButton(self)
        self._toggle_distribution_button.setIcon(toggle_distribution_icon)
        self._toggle_distribution_button.setText(toggle_distribution_text)
        self._toggle_distribution_button.setCheckable(True)
        self._toggle_distribution_button.setStatusTip(toggle_distribution_tip)
        self._toggle_distribution_button.setToolTip(toggle_distribution_tip)
        self._toggle_distribution_button.toggled.connect(self.toggle_distribution_plot)
        self._toggle_distribution_button.toggle()

    def _init_toolbar(self):
        # AI management
        self._main_toolbar = QToolBar("Main toolbar", self)
        self._main_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.addToolBar(Qt.LeftToolBarArea, self._main_toolbar)
        self._main_toolbar.addAction(self._refresh_action)
        self._main_toolbar.addSeparator()
        self._main_toolbar.addWidget(self._toggle_recent_cost_button)
        self._main_toolbar.addWidget(self._toggle_correlation_button)
        self._main_toolbar.addWidget(self._toggle_distribution_button)

    def _init_statusbar(self):
        self._statusbar = self.statusBar()
        self._status_widget = QLabel('Initializing')
        self._statusbar.addPermanentWidget(self._status_widget)

    def update_costs(self):
        self._central_widget.update_cost_plot()

    def update_recent_costs(self):
        self._central_widget.update_metrics()

    def toggle_recent_cost_plot(self, checked):
        self._central_widget.toggle_recent_cost_plot(checked)

    def toggle_correlation_plot(self, checked):
        self._central_widget.toggle_correlation_plot(checked)

    def toggle_distribution_plot(self, checked):
        self._central_widget.toggle_distribution_plot(checked)

    def set_train_running_status(self):
        self._status_widget.setText('Running train')

    def set_train_finished_status(self):
        self._status_widget.setText('Finished train')
