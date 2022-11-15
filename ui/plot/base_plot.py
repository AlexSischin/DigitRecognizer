from pyqtgraph import PlotWidget


class BasePlot(PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._item = self.getPlotItem()
        self.enabled = True
