from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import QMainWindow, QToolBar, QAction, QLabel

import resources.qrc as qrc_resources
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

        self._test_windows = []
        self._selected_ai_duv = None

        self.setWindowTitle('AI trainer')
        self.resize(1000, 600)

        self._init_widgets()
        self._init_actions()
        self._init_toolbar()
        self._init_statusbar()

    def _init_widgets(self):
        # Plots
        self._central_widget = CentralWidget(self._queue)
        self.setCentralWidget(self._central_widget)
        self._central_widget.sigAiVersionSelected.connect(self.on_ai_version_selected)
        self._central_widget.sigTrainRun.connect(self.set_train_running_status)
        self._central_widget.sigTrainFinished.connect(self.set_train_finished_status)

        # Layer combobox widget
        self._layer_widget = LayerWidget()
        self._layer_widget.sigLayerSelected.connect(lambda l: self._central_widget.update_layer(l))
        self._layer_widget.set_layer_count(self._layer_count)

    def _init_actions(self):
        # Launch test window
        test_icon = QIcon(":test-icon")
        test_text = '&Run test'
        test_tip = 'Run test'
        self._test_action = QAction(test_icon, test_text, self)
        self._test_action.setShortcut('Ctrl+L')
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
        self._refresh_action.triggered.connect(self._central_widget.sigRefreshed.emit)

        # Tile widgets
        tile_icon = QIcon(":tile-icon")
        tile_text = '&Tile up widgets'
        tile_tip = 'Tile up widgets'
        self._tile_action = QAction(tile_icon, tile_text, self)
        self._tile_action.setShortcut('Ctrl+T')
        self._tile_action.setStatusTip(tile_tip)
        self._tile_action.setToolTip(tile_tip)
        self._tile_action.triggered.connect(self._central_widget.tile)

        # Toggle recent cost widget
        toggle_recent_cost_icon = QIcon(":recent-cost-plot-icon")
        toggle_recent_cost_text = '&Recent cost plot'
        toggle_recent_cost_tip = 'Open recent cost plot'
        self._toggle_recent_cost_action = QAction(toggle_recent_cost_icon, toggle_recent_cost_text, self)
        self._toggle_recent_cost_action.setStatusTip(toggle_recent_cost_tip)
        self._toggle_recent_cost_action.setToolTip(toggle_recent_cost_tip)
        self._toggle_recent_cost_action.triggered.connect(self._central_widget.add_recent_cost_widget)

        # Toggle correlation widget
        toggle_correlation_icon = QIcon(":correlation-plot-icon")
        toggle_correlation_text = '&Correlation plot'
        toggle_correlation_tip = 'Open correlation plot'
        self._toggle_correlation_action = QAction(toggle_correlation_icon, toggle_correlation_text, self)
        self._toggle_correlation_action.setStatusTip(toggle_correlation_tip)
        self._toggle_correlation_action.setToolTip(toggle_correlation_tip)
        self._toggle_correlation_action.triggered.connect(self._central_widget.add_correlation_widget)

        # Toggle distribution
        toggle_distribution_icon = QIcon(":distribution-plot-icon")
        toggle_distribution_text = '&Distribution plot'
        toggle_distribution_tip = 'Open distribution plot'
        self._toggle_distribution_action = QAction(toggle_distribution_icon, toggle_distribution_text, self)
        self._toggle_distribution_action.setStatusTip(toggle_distribution_tip)
        self._toggle_distribution_action.setToolTip(toggle_distribution_tip)
        self._toggle_distribution_action.triggered.connect(self._central_widget.add_distribution_widget)

        # Toggle gradient length
        toggle_gradient_length_icon = QIcon(":gradient-length-plot-icon")
        toggle_gradient_length_text = '&Gradient length plot'
        toggle_gradient_length_tip = 'Open gradient length plot'
        self._toggle_gradient_length_action = QAction(toggle_gradient_length_icon, toggle_gradient_length_text, self)
        self._toggle_gradient_length_action.setStatusTip(toggle_gradient_length_tip)
        self._toggle_gradient_length_action.setToolTip(toggle_gradient_length_tip)
        self._toggle_gradient_length_action.triggered.connect(self._central_widget.add_grad_len_widget)

        # Toggle weight gradient
        toggle_w_grad_icon = QIcon(":w-grad-plot-icon")
        toggle_w_grad_text = '&Weight gradient plot'
        toggle_w_grad_tip = 'Open weight gradient plot'
        self._toggle_w_grad_action = QAction(toggle_w_grad_icon, toggle_w_grad_text, self)
        self._toggle_w_grad_action.setStatusTip(toggle_w_grad_tip)
        self._toggle_w_grad_action.setToolTip(toggle_w_grad_tip)
        self._toggle_w_grad_action.triggered.connect(self._central_widget.add_w_grad_plot)

        # Toggle bias gradient
        toggle_b_grad_icon = QIcon(":b-grad-plot-icon")
        toggle_b_grad_text = '&Bias gradient plot'
        toggle_b_grad_tip = 'Open bias gradient plot'
        self._toggle_b_grad_action = QAction(toggle_b_grad_icon, toggle_b_grad_text, self)
        self._toggle_b_grad_action.setStatusTip(toggle_b_grad_tip)
        self._toggle_b_grad_action.setToolTip(toggle_b_grad_tip)
        self._toggle_b_grad_action.triggered.connect(self._central_widget.add_b_grad_plot)

    def _init_toolbar(self):
        # AI management
        self._main_toolbar = QToolBar("Main toolbar", self)
        self._main_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.addToolBar(Qt.LeftToolBarArea, self._main_toolbar)

        self._main_toolbar.addAction(self._test_action)

        self._main_toolbar.addSeparator()
        self._main_toolbar.addAction(self._refresh_action)
        self._main_toolbar.addAction(self._tile_action)

        self._main_toolbar.addSeparator()
        self._main_toolbar.addAction(self._toggle_recent_cost_action)
        self._main_toolbar.addAction(self._toggle_correlation_action)
        self._main_toolbar.addAction(self._toggle_distribution_action)
        self._main_toolbar.addAction(self._toggle_gradient_length_action)
        self._main_toolbar.addAction(self._toggle_w_grad_action)
        self._main_toolbar.addAction(self._toggle_b_grad_action)

        self._main_toolbar.addSeparator()
        self._main_toolbar.addWidget(self._layer_widget)

    def _init_statusbar(self):
        self._statusbar = self.statusBar()
        self._status_widget = QLabel('Initializing')
        self._statusbar.addPermanentWidget(self._status_widget)

    def on_ai_test_run(self):
        ai_version = self._selected_ai_duv
        act_funcs = self._activation_functions
        ai_instance = self._central_widget.get_ai_v_by_duv(ai_version, act_funcs)

        test_window = TestWindow(ai_version, ai_instance, f'AI v.{ai_version} test', self._test_data)
        test_window.sigClosed.connect(self.on_test_window_close)
        test_window.show()

        self._test_windows.append(test_window)
        self._update_test_action_state()

    def on_test_window_close(self, window: TestWindow):
        self._test_windows.remove(window)
        self._update_test_action_state()

    def on_ai_version_selected(self, data_used_version):
        self._selected_ai_duv = data_used_version
        self._update_test_action_state()

    def _update_test_action_state(self):
        running_version_tests = [w.ai_version for w in self._test_windows]
        enable_test_action = self._selected_ai_duv and self._selected_ai_duv not in running_version_tests
        self._test_action.setDisabled(not enable_test_action)

    def set_train_running_status(self):
        self._status_widget.setText('Running train')

    def set_train_finished_status(self):
        self._status_widget.setText('Finished train')

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        for window in self._test_windows:
            window.close()
