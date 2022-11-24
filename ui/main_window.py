from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import QMainWindow, QToolBar, QAction, QLabel, QToolButton

import qrc_resources
from ui.central_widget import CentralWidget
from ui.metrics_dispatcher import MetricsDispatchWorkerThread
from ui.test.test_window import TestWindow

# To save from imports optimization by IDEs
qrc_resources = qrc_resources


class MainWindow(QMainWindow):

    def __init__(self, metrics_queue, test_data):
        super().__init__()
        self._queue = metrics_queue
        self._test_data = test_data

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
        self._central_widget = CentralWidget(self._metrics_buff, self)
        self.setCentralWidget(self._central_widget)
        self._central_widget.sigAiVersionSelected.connect(self.on_ai_version_selected)

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

        self._plot_buttons = []
        self._central_widget.sigPlotWidgetsFull.connect(self.disable_inactive_plot_buttons)
        self._central_widget.sigPlotWidgetsAvailable.connect(self.enable_inactive_plot_buttons)

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
        self._plot_buttons.append(self._toggle_recent_cost_button)

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
        self._plot_buttons.append(self._toggle_correlation_button)

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
        self._plot_buttons.append(self._toggle_distribution_button)

        # Toggle gradient length
        toggle_gradient_length_icon = QIcon(":gradient-length-plot-icon")
        toggle_gradient_length_text = '&Gradient length plot'
        toggle_gradient_length_tip = 'Toggle gradient length plot'
        self._toggle_gradient_length_button = QToolButton(self)
        self._toggle_gradient_length_button.setIcon(toggle_gradient_length_icon)
        self._toggle_gradient_length_button.setText(toggle_gradient_length_text)
        self._toggle_gradient_length_button.setCheckable(True)
        self._toggle_gradient_length_button.setStatusTip(toggle_gradient_length_tip)
        self._toggle_gradient_length_button.setToolTip(toggle_gradient_length_tip)
        self._toggle_gradient_length_button.toggled.connect(self.toggle_gradient_length_plot)
        self._plot_buttons.append(self._toggle_gradient_length_button)

        # Toggle buttons
        self._toggle_recent_cost_button.toggle()
        self._toggle_correlation_button.toggle()
        self._toggle_distribution_button.toggle()

    def _init_toolbar(self):
        # AI management
        self._main_toolbar = QToolBar("Main toolbar", self)
        self._main_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.addToolBar(Qt.LeftToolBarArea, self._main_toolbar)

        self._main_toolbar.addAction(self._test_action)
        self._main_toolbar.addSeparator()
        self._main_toolbar.addAction(self._refresh_action)
        self._main_toolbar.addSeparator()
        self._main_toolbar.addWidget(self._toggle_recent_cost_button)
        self._main_toolbar.addWidget(self._toggle_correlation_button)
        self._main_toolbar.addWidget(self._toggle_distribution_button)
        self._main_toolbar.addWidget(self._toggle_gradient_length_button)

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
        enable_test_button = self._selected_ai_version \
                             and self._selected_ai_version not in running_version_tests
        self._test_action.setDisabled(not enable_test_button)

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

    def disable_inactive_plot_buttons(self):
        for b in self._plot_buttons:
            if not b.isChecked():
                b.setDisabled(True)

    def enable_inactive_plot_buttons(self):
        for b in self._plot_buttons:
            if not b.isChecked():
                b.setDisabled(False)

    def set_train_running_status(self):
        self._status_widget.setText('Running train')

    def set_train_finished_status(self):
        self._status_widget.setText('Finished train')

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        for window in self._test_windows:
            window.close()
