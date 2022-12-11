import multiprocessing as mp
import random
from typing import Iterable, Generator, TypeVar

import numpy as np
from PyQt5.QtWidgets import QApplication

import ai
from batched_queue import BufferedQueue
from ui.main_window import MainWindow

T = TypeVar('T')


def random_iterator(values: list[T]) -> Generator[T, None, None]:
    values_copy = values.copy()
    random.shuffle(values_copy)
    yield from values_copy


def random_extended_chunked_list(values: list[T], chunk_size: int, chunk_count: int) -> list[list[T]]:
    result_list = []
    it = iter([])
    while len(result_list) < chunk_count:
        chunk = []
        while len(chunk) < chunk_size:
            try:
                v = next(it)
            except StopIteration:
                it = random_iterator(values)
                v = next(it)
            chunk.append(v)
        result_list.append(chunk)

    return result_list


def train(queue, train_data, ai_model):
    for xy_chunk in train_data:
        xs, ys = zip(*xy_chunk)
        xy_chunk.clear()
        metric = ai_model.train(xs, ys)
        queue.put_nowait(metric)
    queue.put_nowait(None)


class AiTrainer(QApplication):
    def __init__(self,
                 ai_model: ai.Ai,
                 train_data: Iterable[list[tuple[np.ndarray, np.ndarray]]],
                 test_data: Iterable[tuple[np.ndarray, np.ndarray]],
                 queue_max_size=3,
                 queue_batch_size=5
                 ) -> None:
        super().__init__([])

        queue = BufferedQueue(max_size=queue_max_size, batch_size=queue_batch_size)
        self._window = MainWindow(queue, test_data, ai_model)
        self._train_process = mp.Process(target=train, args=(queue, train_data, ai_model), daemon=True)

    def exec(self) -> int:
        self._train_process.start()
        self._window.show()
        return super().exec()
