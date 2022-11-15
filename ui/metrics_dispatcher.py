from PyQt5.QtCore import QThread, pyqtSignal


class MetricsDispatchWorkerThread(QThread):
    started = pyqtSignal()
    updated = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, metrics_queue, metrics_buff) -> None:
        super().__init__()
        self._metrics_queue = metrics_queue
        self._metrics_buff = metrics_buff

    def run(self) -> None:
        self.started.emit()
        while True:
            metrics_batch = self._metrics_queue.get(block=True)
            if metrics_batch:
                self._metrics_buff.extend(metrics_batch)
                self.updated.emit()
            else:
                break
        self.finished.emit()