import ai
from ui.metrics_dispatcher import Hub


class AiHub(Hub):

    def __init__(self) -> None:
        self._ai_version = []

    def update_data(self, metrics: list[ai.TrainMetric]):
        self._ai_version.extend([(m.w, m.b) for m in metrics])

    def get_version(self, v):
        return self._ai_version[v]
