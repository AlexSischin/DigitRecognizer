import dataclasses

import numpy as np
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QVBoxLayout, QSizePolicy, QFormLayout, \
    QHBoxLayout

import ai
import qrc_resources
from ui.test.dataset.img_viewer import ImageViewer
from utils.zip_utils import zip2

# To save from imports optimization by IDEs
qrc_resources = qrc_resources


@dataclasses.dataclass
class TestInfo:
    test_count = 0
    error_count = 0
    accuracy = None
    digit_img = None
    guess = None

    def update(self, digit_img, y_act, y_exp):
        self.test_count += 1
        self.error_count += 0 if y_exp == y_act else 1
        self.accuracy = 1 - self.error_count / self.test_count
        self.digit_img = digit_img
        self.guess = y_act


class DatasetTestWidget(QWidget):
    def __init__(self, ai_model: ai.Ai, test_data):
        super().__init__()
        self._ai_model = ai_model
        self._test_data = test_data
        self._default_interval = 0

        self._init_layout()
        self._init_test_timer()
        self.reset_test_info()

        self.set_interval()

    def _init_layout(self):
        layout = QHBoxLayout()
        layout.addLayout(self._create_panel_layout())
        layout.addWidget(self._create_image_viewer())
        self.setLayout(layout)

    def _create_panel_layout(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        layout.addWidget(self._create_buttons_widget(), alignment=Qt.AlignCenter)
        layout.addWidget(self._create_parameters_widget(), alignment=Qt.AlignCenter)
        layout.addWidget(self._create_indicator_widget(), alignment=Qt.AlignCenter)
        return layout

    def _create_buttons_widget(self):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout = QHBoxLayout()

        self._run_button = QPushButton('Run test')
        self._run_button.setFixedSize(QSize(70, 30))
        self._run_button.clicked.connect(self.start_test)
        layout.addWidget(self._run_button)

        self._pause_button = QPushButton('Pause test')
        self._pause_button.setFixedSize(QSize(70, 30))
        self._pause_button.clicked.connect(self.pause_test)
        self._pause_button.setDisabled(True)
        layout.addWidget(self._pause_button)

        self._reset_button = QPushButton('Reset test')
        self._reset_button.setFixedSize(QSize(70, 30))
        self._reset_button.clicked.connect(self.reset_test)
        self._reset_button.setDisabled(True)
        layout.addWidget(self._reset_button)

        widget.setLayout(layout)
        return widget

    def _create_parameters_widget(self):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout = QFormLayout()

        self._interval_edit = QLineEdit()
        self._interval_edit.setFixedSize(QSize(100, 20))
        self._interval_edit.setValidator(QIntValidator(0, 10000))
        self._interval_edit.setText(str(self._default_interval))
        self._interval_edit.editingFinished.connect(self.set_interval)
        layout.addRow('Interval (ms)', self._interval_edit)

        widget.setLayout(layout)
        return widget

    def _create_indicator_widget(self):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout = QFormLayout(self)

        self._test_count_edit = QLineEdit()
        self._test_count_edit.setFixedSize(QSize(100, 20))
        self._test_count_edit.setReadOnly(True)
        layout.addRow('Test count', self._test_count_edit)

        self._error_count_edit = QLineEdit()
        self._error_count_edit.setFixedSize(QSize(100, 20))
        self._error_count_edit.setReadOnly(True)
        layout.addRow('Error count', self._error_count_edit)

        self._accuracy_edit = QLineEdit()
        self._accuracy_edit.setFixedSize(QSize(100, 20))
        self._accuracy_edit.setReadOnly(True)
        layout.addRow('Accuracy', self._accuracy_edit)

        self._guess_edit = QLineEdit()
        self._guess_edit.setFixedSize(QSize(20, 20))
        self._guess_edit.setReadOnly(True)
        layout.addRow('Guess', self._guess_edit)

        widget.setLayout(layout)
        return widget

    def _create_image_viewer(self):
        self._image_viewer = ImageViewer()
        return self._image_viewer

    def _init_test_timer(self):
        self._test_timer = QTimer(self)
        self._test_timer.timeout.connect(self.update_test_info)

    def start_test(self):
        self._test_timer.start()

        self._run_button.setDisabled(True)
        self._pause_button.setDisabled(False)
        self._reset_button.setDisabled(False)

    def pause_test(self):
        self._test_timer.stop()

        self._run_button.setDisabled(False)
        self._pause_button.setDisabled(True)

    def finish_test(self):
        self._test_timer.stop()

        self._run_button.setDisabled(True)
        self._pause_button.setDisabled(True)
        self._reset_button.setDisabled(False)

    def reset_test(self):
        self.reset_test_info()
        self._test_timer.stop()

        self._run_button.setDisabled(False)
        self._pause_button.setDisabled(True)
        self._reset_button.setDisabled(True)

    def set_interval(self):
        self._test_timer.setInterval(int(self._interval_edit.text()))

    def reset_test_info(self):
        self._data_iterator = iter(zip2(*self._test_data))
        self._test_info = TestInfo()
        self._display_info()

    def update_test_info(self):
        try:
            x_test, y_exp = next(self._data_iterator)
        except StopIteration:
            self.finish_test()
            return
        x_test_vec = x_test.flatten() / 255
        y_act_vec = self._ai_model.feed(x_test_vec)
        y_act = np.argmax(y_act_vec)
        self._test_info.update(x_test, y_act, y_exp)
        self._display_info()

    def _display_info(self):
        info = self._test_info
        self._test_count_edit.setText(str(info.test_count))
        self._error_count_edit.setText(str(info.error_count))
        self._accuracy_edit.setText('{:.3}'.format(info.accuracy) if info.accuracy is not None else '')
        self._guess_edit.setText(str(info.guess if info.guess is not None else ''))
        self._image_viewer.set_image(info.digit_img)
