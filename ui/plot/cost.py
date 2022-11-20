import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal

from ui.metrics_dispatcher import TrainMetric
from ui.plot.base_plot import BasePlot


class CostPlot(BasePlot):
    sigSpotSelected = pyqtSignal(float)

    def __init__(self) -> None:
        super().__init__()

        self.setStyleSheet("border: 1px solid black;")
        self.setTitle("Cost by train data")

        self.setLabel('left', "Cost")
        self.setLabel('bottom', "Train data")
        self.curve = self.plot(pen=(0, 128, 0), symbol='t', symbolBrush=(0, 128, 0))
        self.curve.scatter.opts['hoverable'] = True
        self.curve.scatter.opts['clickable'] = True
        self.data = np.random.normal(size=(10, 1000))
        self.showGrid(x=True, y=True)

        self.lr = pg.LinearRegionItem((0, 0))
        self.lr.setZValue(-10)
        self.lr.hide()
        self.addItem(self.lr)

        self.hovered_spot = None
        self.selected_spot = None
        self.hover_scatter = pg.ScatterPlotItem(brush=pg.mkBrush(217, 83, 25), symbol='t')
        self.selection_scatter = pg.ScatterPlotItem(size=11, brush=pg.mkBrush(217, 83, 25), symbol='s')
        self.addItem(self.hover_scatter)
        self.addItem(self.selection_scatter)

        self.curve.sigPointsHovered.connect(self.handle_points_hover)
        self.curve.sigPointsClicked.connect(self.handle_points_click)
        self.hover_scatter.sigClicked.connect(self.handle_points_click)

    def set_data(self, metrics: list[TrainMetric]):
        nodes = [(m.data_used, m.cost) for m in metrics]
        self.curve.setData(np.array(nodes))
        if metrics:
            self.lr.setBounds((metrics[0].data_used, metrics[-1].data_used))
            if not self.lr.isVisible():
                self.lr.show()
                self.lr.setRegion((metrics[0].data_used, metrics[min(5, len(metrics) - 1)].data_used))
        else:
            self.lr.hide()

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
