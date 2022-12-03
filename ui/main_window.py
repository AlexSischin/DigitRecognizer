from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import QMainWindow, QToolBar, QAction, QLabel

import qrc_resources
from ui.central_widget import CentralWidget
from ui.layer_widget import LayerWidget
from ui.metrics_dispatcher import MetricsDispatchWorkerThread
from ui.test.test_window import TestWindow

# To save from imports optimization by IDEs
qrc_resources = qrc_resources


class MainWindow(QMainWindow):

    def __init__(self, metrics_queue, test_data, layer_count, activation_functions=None):
        super().__init__()
        self._queue = metrics_queue
        self._test_data = test_data
        self._layer_count = layer_count
        self._activation_functions = activation_functions

        self._metrics_buff = []
        self._test_windows = []
        self._selected_ai_version = None

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
        self._central_widget = CentralWidget(self._metrics_buff, self._activation_functions)
        self.setCentralWidget(self._central_widget)
        self._central_widget.sigAiVersionSelected.connect(self.on_ai_version_selected)

        # Layer combobox widget
        self._layer_widget = LayerWidget()
        self._layer_widget.sigLayerSelected.connect(self.select_layer)
        self._layer_widget.set_layer_count(self._layer_count)

    def _init_actions(self):
        # Launch test window
        test_icon = QIcon(":test-icon")
        test_text = '&Run test'
        test_tip = 'Run test'
        self._test_action = QAction(test_icon, test_text, self)
        self._test_action.setShortcuts((QKeySequence.Refresh, 'Ctrl+L'))
        self._test_action.setStatusTip(test_tip)
        self._test_action.setToolTip(test_tip)
        self._test_action.triggered.connect(self.on_ai_test_run)
        self._update_test_action_state()

        # Refresh cost plot
        refresh_icon = QIcon(":refresh-icon")
        refresh_text = '&Refresh costs'
        refresh_tip = 'Refresh costs plot'
        self._refresh_action = QAction(refresh_icon, refresh_text, self)
        self._refresh_action.setShortcuts((QKeySequence.Refresh, 'Ctrl+R'))
        self._refresh_action.setStatusTip(refresh_tip)
        self._refresh_action.setToolTip(refresh_tip)
        self._refresh_action.triggered.connect(self.update_costs)

        self._plot_actions = []
        self._central_widget.sigPlotWidgetsFull.connect(self.disable_inactive_plot_actions)
        self._central_widget.sigPlotWidgetsAvailable.connect(self.enable_inactive_plot_actions)

        # Toggle distribution
        toggle_recent_cost_icon = QIcon(":recent-cost-plot-icon")
        toggle_recent_cost_text = '&Recent cost plot'
        toggle_recent_cost_tip = 'Toggle recent cost plot'
        self._toggle_recent_cost_action = QAction(toggle_recent_cost_icon, toggle_recent_cost_text, self)
        self._toggle_recent_cost_action.setCheckable(True)
        self._toggle_recent_cost_action.setStatusTip(toggle_recent_cost_tip)
        self._toggle_recent_cost_action.setToolTip(toggle_recent_cost_tip)
        self._toggle_recent_cost_action.toggled.connect(self.toggle_recent_cost_plot)
        self._plot_actions.append(self._toggle_recent_cost_action)

        # Toggle correlation
        toggle_correlation_icon = QIcon(":correlation-plot-icon")
        toggle_correlation_text = '&Correlation plot'
        toggle_correlation_tip = 'Toggle correlation plot'
        self._toggle_correlation_action = QAction(toggle_correlation_icon, toggle_correlation_text, self)
        self._toggle_correlation_action.setCheckable(True)
        self._toggle_correlation_action.setStatusTip(toggle_correlation_tip)
        self._toggle_correlation_action.setToolTip(toggle_correlation_tip)
        self._toggle_correlation_action.toggled.connect(self.toggle_correlation_plot)
        self._plot_actions.append(self._toggle_correlation_action)

        # Toggle distribution
        toggle_distribution_icon = QIcon(":distribution-plot-icon")
        toggle_distribution_text = '&Distribution plot'
        toggle_distribution_tip = 'Toggle distribution plot'
        self._toggle_distribution_action = QAction(toggle_distribution_icon, toggle_distribution_text, self)
        self._toggle_distribution_action.setCheckable(True)
        self._toggle_distribution_action.setStatusTip(toggle_distribution_tip)
        self._toggle_distribution_action.setToolTip(toggle_distribution_tip)
        self._toggle_distribution_action.toggled.connect(self.toggle_distribution_plot)
        self._plot_actions.append(self._toggle_distribution_action)

        # Toggle gradient length
        toggle_gradient_length_icon = QIcon(":gradient-length-plot-icon")
        toggle_gradient_length_text = '&Gradient length plot'
        toggle_gradient_length_tip = 'Toggle gradient length plot'
        self._toggle_gradient_length_action = QAction(toggle_gradient_length_icon, toggle_gradient_length_text, self)
        self._toggle_gradient_length_action.setCheckable(True)
        self._toggle_gradient_length_action.setStatusTip(toggle_gradient_length_tip)
        self._toggle_gradient_length_action.setToolTip(toggle_gradient_length_tip)
        self._toggle_gradient_length_action.toggled.connect(self.toggle_gradient_length_plot)
        self._plot_actions.append(self._toggle_gradient_length_action)

        # Toggle gradient
        toggle_w_grad_icon = QIcon(":w-grad-plot-icon")
        toggle_w_grad_text = '&Weight gradient plot'
        toggle_w_grad_tip = 'Toggle weight gradient plot'
        self._toggle_w_grad_action = QAction(toggle_w_grad_icon, toggle_w_grad_text, self)
        self._toggle_w_grad_action.setCheckable(True)
        self._toggle_w_grad_action.setStatusTip(toggle_w_grad_tip)
        self._toggle_w_grad_action.setToolTip(toggle_w_grad_tip)
        self._toggle_w_grad_action.toggled.connect(self.toggle_w_grad_plot)
        self._plot_actions.append(self._toggle_w_grad_action)

        # Toggle actions
        self._toggle_recent_cost_action.toggle()
        self._toggle_correlation_action.toggle()
        self._toggle_distribution_action.toggle()

    def _init_toolbar(self):
        # AI management
        self._main_toolbar = QToolBar("Main toolbar", self)
        self._main_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.addToolBar(Qt.LeftToolBarArea, self._main_toolbar)

        self._main_toolbar.addAction(self._test_action)

        self._main_toolbar.addSeparator()
        self._main_toolbar.addAction(self._refresh_action)

        self._main_toolbar.addSeparator()
        self._main_toolbar.addAction(self._toggle_recent_cost_action)
        self._main_toolbar.addAction(self._toggle_correlation_action)
        self._main_toolbar.addAction(self._toggle_distribution_action)
        self._main_toolbar.addAction(self._toggle_gradient_length_action)
        self._main_toolbar.addAction(self._toggle_w_grad_action)

        self._main_toolbar.addSeparator()
        self._main_toolbar.addWidget(self._layer_widget)

    def _init_statusbar(self):
        self._statusbar = self.statusBar()
        self._status_widget = QLabel('Initializing')
        self._statusbar.addPermanentWidget(self._status_widget)

    def on_ai_test_run(self):
        ai_version = self._selected_ai_version
        ai_instance = self._central_widget.get_ai_version(ai_version)

        test_window = TestWindow(ai_version, ai_instance, f'AI v.{ai_version} test', self._test_data)
        test_window.sigClosed.connect(self.on_test_window_close)
        test_window.show()

        self._test_windows.append(test_window)
        self._update_test_action_state()

    def on_test_window_close(self, window: TestWindow):
        self._test_windows.remove(window)
        self._update_test_action_state()

    def on_ai_version_selected(self, version_id):
        self._selected_ai_version = version_id
        self._update_test_action_state()

    def _update_test_action_state(self):
        running_version_tests = [w.ai_version for w in self._test_windows]
        enable_test_action = self._selected_ai_version \
                             and self._selected_ai_version not in running_version_tests
        self._test_action.setDisabled(not enable_test_action)

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

    def toggle_gradient_length_plot(self, checked):
        self._central_widget.toggle_gradient_length_plot(checked)

    def toggle_w_grad_plot(self, checked):
        self._central_widget.toggle_w_grad_plot(checked)

    def disable_inactive_plot_actions(self):
        for b in self._plot_actions:
            if not b.isChecked():
                b.setDisabled(True)

    def enable_inactive_plot_actions(self):
        for b in self._plot_actions:
            if not b.isChecked():
                b.setDisabled(False)

    def set_train_running_status(self):
        self._status_widget.setText('Running train')

    def set_train_finished_status(self):
        self._status_widget.setText('Finished train')

    def select_layer(self, layer):
        self._central_widget.set_layer(layer)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        for window in self._test_windows:
            window.close()
