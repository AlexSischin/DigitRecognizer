import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtWidgets import QLabel, QSizePolicy


class ImageViewer(QLabel):
    def __init__(self) -> None:
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)
        self.resize(200, 200)

        self.set_image(None)

    def set_image(self, mono_img: np.ndarray):
        if mono_img is None:
            pixmap = QPixmap(1, 1)
            pixmap.fill(QColor('gray'))
        else:
            data = mono_img.data
            height, width = mono_img.shape
            img = QImage(data, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(img)
        self._update_pixmap(pixmap)

    def resizeEvent(self, a0: QtGui.QResizeEvent):
        self._update_pixmap(self.pixmap)

    def _update_pixmap(self, pixmap):
        self.pixmap = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
        self.setPixmap(self.pixmap)
