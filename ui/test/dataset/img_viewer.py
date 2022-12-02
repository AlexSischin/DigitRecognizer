import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtWidgets import QLabel, QSizePolicy


class ImageViewer(QLabel):
    def __init__(self, data_type=np.uint8, image_data_format=QImage.Format_Grayscale8) -> None:
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)
        self.resize(200, 200)
        self._data_type = data_type
        self._image_data_format = image_data_format

        self.set_image(None)

    def set_image(self, data: np.ndarray | None):
        if data is None:
            pixmap = QPixmap(1, 1)
            pixmap.fill(QColor('gray'))
        else:
            if not isinstance(data, np.ndarray):
                raise ValueError(f'Data must be numpy.ndarray or None type. Given: {type(data)}')
            if data.ndim != 2:
                raise ValueError(f'Data must have two dimensions. Given: {data.ndim}')
            if data.dtype != self._data_type:
                data = data.astype(self._data_type)
            height, width = data.shape
            img = QImage(data, height, width, self._image_data_format)
            pixmap = QPixmap.fromImage(img)
        self._update_pixmap(pixmap)

    def resizeEvent(self, a0: QtGui.QResizeEvent):
        self._update_pixmap(self.pixmap)

    def _update_pixmap(self, pixmap):
        self.pixmap = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
        self.setPixmap(self.pixmap)
