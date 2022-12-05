from typing import Iterable

from PyQt5.QtCore import QThread, pyqtSignal

import ai


class Hub:
    def update_data(self, metrics: list[ai.TrainMetric]):
        pass


class MetricsDispatchWorkerThread(QThread):
    started = pyqtSignal()
    updated = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, metrics_queue, hubs: Iterable[Hub] | None = None) -> None:
        super().__init__()
        if hubs is None:
            hubs = []
        self._queue = metrics_queue
        self._hubs = hubs

    def run(self) -> None:
        self.started.emit()
        while True:
            metrics_batch: list[ai.TrainMetric] = self._queue.get(block=True)
            if metrics_batch:
                for hub in self._hubs:
                    hub.update_data(metrics_batch)
                self.updated.emit()
            else:
                break
        self.finished.emit()
