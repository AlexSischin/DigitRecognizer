from enum import Enum

import ai
from ui.metrics_dispatcher import Hub


class Approx(Enum):
    EQ = 1
    LE = 2
    GE = 3


class VersionHub(Hub):
    def __init__(self) -> None:
        super().__init__()
        self._data_used_versions = []

    def get_version(self, duv, approx: Approx | None = None):
        if approx is None:
            approx = Approx.EQ
        if approx is Approx.EQ:
            return self._data_used_versions.index(duv)
        elif approx is Approx.LE:
            near_vs = [v for v in self._data_used_versions if v <= duv]
            near_v = max(near_vs) if len(near_vs) else None
            return self._data_used_versions.index(near_v) if near_v else None
        elif approx is Approx.GE:
            near_vs = [v for v in self._data_used_versions if v >= duv]
            near_v = min(near_vs) if len(near_vs) else None
            return self._data_used_versions.index(near_v) if near_v else None
        raise ValueError(f'Invalid approx parameter value: {approx}')

    def update_data(self, metrics: list[ai.TrainMetric]):
        self._data_used_versions.extend([m.data_used for m in metrics])
