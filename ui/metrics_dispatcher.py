from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

import ai


@dataclass(frozen=True)
class TrainMetric:
    w_gradient: list[np.ndarray]
    b_gradient: list[np.ndarray]
    gradient_len: float
    costs: list[np.ndarray]
    cost: float
    inputs: list[np.ndarray]
    outputs: list[np.ndarray]
    expected: list[np.ndarray]
    data_used: int


class MetricsDispatchWorkerThread(QThread):
    started = pyqtSignal()
    updated = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, metrics_queue, metrics_buff) -> None:
        super().__init__()
        self._metrics_queue = metrics_queue
        self._metrics_buff = metrics_buff
        self._data_used_buff = []
        self._metrics_used = 0

    def run(self) -> None:
        self.started.emit()
        while True:
            metrics_batch: list[ai.TrainMetric] = self._metrics_queue.get(block=True)
            if metrics_batch:
                extended_metrics_batch = []
                for metric in metrics_batch:
                    self._metrics_used += len(metric.inputs)
                    extended_metric = TrainMetric(
                        w_gradient=metric.w_gradient, b_gradient=metric.b_gradient, gradient_len=metric.gradient_len,
                        costs=metric.costs, cost=metric.cost, inputs=metric.inputs, outputs=metric.outputs,
                        expected=metric.expected, data_used=self._metrics_used
                    )
                    extended_metrics_batch.append(extended_metric)
                self._metrics_buff.extend(extended_metrics_batch)
                self.updated.emit()
            else:
                break
        self.finished.emit()
