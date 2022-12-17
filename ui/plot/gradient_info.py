from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QGridLayout, QSizePolicy, QLabel


def _create_label(text=''):
    label = QLabel(text)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


class GradientInfo(QWidget):
    def __init__(self):
        super().__init__()

        self._empty_text = '-'

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._layout = QGridLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._row_count = 0

        self._layer_size_label = _create_label(self._empty_text)
        self.add_labeled_widget('Layer size: ', self._layer_size_label)

        self._layer_mean_label = _create_label(self._empty_text)
        self.add_labeled_widget('Layer mean: ', self._layer_mean_label)

        self._layer_sd_label = _create_label(self._empty_text)
        self.add_labeled_widget('Layer SD: ', self._layer_sd_label)

        self._size_label = _create_label(self._empty_text)
        self.add_labeled_widget('Size: ', self._size_label)

        self._mean_label = _create_label(self._empty_text)
        self.add_labeled_widget('Mean: ', self._mean_label)

        self._sd_label = _create_label(self._empty_text)
        self.add_labeled_widget('SD: ', self._sd_label)

        self.setLayout(self._layout)

    def add_labeled_widget(self, label: str, widget: QWidget):
        self._layout.addWidget(QLabel(label), self._row_count, 0)
        self._layout.addWidget(widget, self._row_count, 1)
        self._row_count += 1

    def set_layer_info(self, size, mean, sd):
        self._layer_size_label.setText(f'{size:g}')
        self._layer_mean_label.setText(f'{mean:.2e}')
        self._layer_sd_label.setText(f'{sd:.2e}')

    def set_info(self, size, mean, sd):
        self._size_label.setText(f'{size:g}')
        self._mean_label.setText(f'{mean:.2e}')
        self._sd_label.setText(f'{sd:.2e}')

    def clear_layer_info(self):
        self._layer_size_label.setText(self._empty_text)
        self._layer_mean_label.setText(self._empty_text)
        self._layer_sd_label.setText(self._empty_text)

    def clear_info(self):
        self._size_label.setText(self._empty_text)
        self._mean_label.setText(self._empty_text)
        self._sd_label.setText(self._empty_text)
