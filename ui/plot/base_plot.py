from pyqtgraph import PlotWidget


class BasePlot(PlotWidget):
    def __init__(self):
        super().__init__()
        self._item = self.getPlotItem()
        self.enabled = True
