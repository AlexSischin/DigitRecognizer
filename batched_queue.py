import multiprocessing as mp


class BufferedQueue:
    def __init__(self, max_size, batch_size, send_incomplete=True) -> None:
        super().__init__()
        self._queue = mp.Queue(maxsize=max_size)
        self._batch = []
        self._batch_size = batch_size
        self._send_incomplete = send_incomplete

    def put_nowait(self, o):
        if o is None:
            if self._send_incomplete and len(self._batch) > 0:
                self._queue.put_nowait(tuple(self._batch))
            self._queue.put_nowait(None)
            return

        self._batch.append(o)
        if len(self._batch) >= self._batch_size:
            self._queue.put_nowait(tuple(self._batch))
            self._batch.clear()

    def get(self, block=True, timeout=None) -> tuple:
        return self._queue.get(block, timeout)
