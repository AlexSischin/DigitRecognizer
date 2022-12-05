import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from pyqtgraph import PlotWidget

import ai
from ui.metrics_dispatcher import Hub


class CostHub(Hub):
    def __init__(self) -> None:
        super().__init__()
        self._costs = []

    def update_data(self, metrics: list[ai.TrainMetric]):
        self._costs.extend([(m.data_used, m.cost) for m in metrics])

    def calc(self):
        return np.array(self._costs)


class CostWidget(QWidget):
    sigSpotSelected = pyqtSignal(float)
    sigRegionChanged = pyqtSignal(float, float)

    def __init__(self, hub: CostHub) -> None:
        super().__init__()

        self._hub = hub

        self.setMinimumHeight(100)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        plot = PlotWidget()
        plot.setTitle("Cost by train data")
        plot.setLabel('left', "Cost")
        plot.setLabel('bottom', "Train data")
        plot.showGrid(x=True, y=True)

        self.curve = plot.plot(pen=(0, 128, 0), symbol='t', symbolBrush=(0, 128, 0))
        self.curve.scatter.opts['hoverable'] = True
        self.curve.scatter.opts['clickable'] = True
        self.data = np.random.normal(size=(10, 1000))

        self._lr = pg.LinearRegionItem((0, 0))
        self._lr.setZValue(-10)
        self._lr.hide()
        self._lr.sigRegionChanged.connect(lambda r: self.sigRegionChanged.emit(*self._lr.getRegion()))
        plot.addItem(self._lr)

        self.hovered_spot = None
        self.selected_spot = None
        self.hover_scatter = pg.ScatterPlotItem(brush=pg.mkBrush(217, 83, 25), symbol='t')
        self.selection_scatter = pg.ScatterPlotItem(size=11, brush=pg.mkBrush(217, 83, 25), symbol='s')
        plot.addItem(self.hover_scatter)
        plot.addItem(self.selection_scatter)

        self.curve.sigPointsHovered.connect(self.handle_points_hover)
        self.curve.sigPointsClicked.connect(self.handle_points_click)
        self.hover_scatter.sigClicked.connect(self.handle_points_click)

        layout.addWidget(plot)
        self.setLayout(layout)

    def refresh(self):
        nodes = self._hub.calc()
        self.curve.setData(nodes)
        if nodes.size > 0:
            self._lr.setBounds((nodes[0, 0], nodes[-1, 0]))
            if not self._lr.isVisible():
                self._lr.show()
                region_left = nodes[0, 0]
                region_right = nodes[min(5, nodes.shape[0] - 1), 0]
                self._lr.setRegion((region_left, region_right))
        else:
            self._lr.hide()

    def handle_points_hover(self, _data_item, spots, _ev):
        if len(spots):
            self.hover_spot(spots[0].pos())
        else:
            self.hover_spot(spots[None])

    def hover_spot(self, spot):
        if spot == self.selected_spot:
            spot = None
        if self.hovered_spot != spot:
            self.hovered_spot = spot
            if spot:
                x, y = spot
                self.hover_scatter.setData(x=[x], y=[y])
            else:
                self.hover_scatter.setData(x=[], y=[])

    def handle_points_click(self, _data_item, spots, ev):
        if len(spots):
            ev.accept()
            self.select_spot(spots[0].pos())
        else:
            self.select_spot(None)

    def select_spot(self, spot):
        if self.selected_spot != spot:
            self.selected_spot = spot
            if spot:
                x, y = spot
                self.selection_scatter.setData(x=[x], y=[y])
                self.sigSpotSelected.emit(x)
            else:
                self.selection_scatter.setData(x=[], y=[])
                self.sigSpotSelected.emit(None)
