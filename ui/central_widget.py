from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QWidget

from ui.metrics_dispatcher import TrainMetric
from ui.plot.correlation import CorrelationPlot
from ui.plot.cost import CostPlot
from ui.plot.distribution import DistributionPlot
from ui.plot.gradient_length_plot import GradientLengthPlot
from ui.plot.recent_cost import RecentCostPlot


class CentralWidget(QWidget):
    plot_widgets_full = pyqtSignal()
    plot_widgets_available = pyqtSignal()

    def __init__(self, metrics_buff: list[TrainMetric], parent=None):
        super().__init__(parent)
        self._metrics_buff = metrics_buff
        self._last_region = None
        self._row_count = 2
        self._column_count = 3

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet('background-color: black;')

        self._init_plots()

    def _init_plots(self):
        self._layout = QGridLayout(self)
        self._layout.setColumnStretch(0, 1)
        self._layout.setColumnStretch(1, 1)
        self._layout.setColumnStretch(2, 1)
        self._layout.setRowStretch(0, 1)
        self._layout.setRowStretch(1, 1)
        self.setLayout(self._layout)

        self._optional_plots = []

        # All costs
        self._cost_plot = CostPlot()
        self._cost_plot.lr.sigRegionChanged.connect(self.update_region)
        self._cost_plot.sigSpotSelected.connect(self.select_spot)
        self._layout.addWidget(self._cost_plot, 0, 0, 1, 3)

        # Recent costs
        self._recent_cost_plot = RecentCostPlot()
        self._optional_plots.append(self._recent_cost_plot)

        # Correlation
        self._correlation_plot = CorrelationPlot()
        self._optional_plots.append(self._correlation_plot)

        # Distribution
        self._distribution_plot = DistributionPlot()
        self._optional_plots.append(self._distribution_plot)

        # Gradient length
        self._gradient_length_plot = GradientLengthPlot()
        self._optional_plots.append(self._gradient_length_plot)

        for p in self._optional_plots:
            p.hide()
            p.enabled = False

        self._plot_widgets_available = (self._row_count - 1) * self._column_count
        self._plot_widgets_active = 0

    def update_region(self):
        left_bound, right_bound = self._cost_plot.lr.getRegion()
        region_metrics = [m for m in self._metrics_buff if left_bound <= m.data_used <= right_bound]
        region = (region_metrics[0].data_used, region_metrics[-1].data_used) if region_metrics else None

        if region != self._last_region:
            self._last_region = region
            self._correlation_plot.set_data(region_metrics)
            self._distribution_plot.set_data(region_metrics)
            self._gradient_length_plot.set_data(region_metrics)

    def update_cost_plot(self):
        self._cost_plot.set_data(self._metrics_buff)

    def update_metrics(self):
        recent_metrics = self._metrics_buff[-50:]
        self._recent_cost_plot.set_data(recent_metrics)

    def select_spot(self, data_used):
        metric = [m for m in self._metrics_buff if m.data_used == data_used][0]
        print(f'Selected metric: {(metric.data_used, metric.cost)}')

    def rearrange_plots(self):
        counter = 0
        for i, pw in enumerate(self._optional_plots):
            self._layout.removeWidget(pw)
            pw.hide()
            if pw.enabled:
                r, c = divmod(counter + self._column_count, self._column_count)
                if r >= self._row_count or c >= self._column_count:
                    raise ValueError('Out of grid bounds')
                self._layout.addWidget(pw, r, c)
                pw.show()
                counter += 1

    def _toggle_plot(self, plot, checked):
        if plot.enabled == checked:
            raise ValueError(f'Plot widget is already in the desired state')
        if checked:
            self._plot_widgets_active += 1
            if self._plot_widgets_active >= self._plot_widgets_available:
                self.plot_widgets_full.emit()
        else:
            if self._plot_widgets_active == self._plot_widgets_available:
                self.plot_widgets_available.emit()
            self._plot_widgets_active -= 1
        plot.enabled = checked
        self.rearrange_plots()

    def toggle_recent_cost_plot(self, checked):
        self._toggle_plot(self._recent_cost_plot, checked)

    def toggle_correlation_plot(self, checked):
        self._toggle_plot(self._correlation_plot, checked)

    def toggle_distribution_plot(self, checked):
        self._toggle_plot(self._distribution_plot, checked)

    def toggle_gradient_length_plot(self, checked):
        self._toggle_plot(self._gradient_length_plot, checked)
