import multiprocessing as mp
import statistics
import sys

import numpy as np
import pyqtgraph as pg
import tensorflow as tf
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QFile, QTextStream
from PyQt5.QtWidgets import QApplication

import ai
from utils import iter_utils as iu
from utils import zip_utils as zu
from utils.time_utils import TimeLog
from ui.main_window import MainWindow
import qrc_resources

# To save from imports optimization by IDEs
qrc_resources = qrc_resources

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
layer_sizes = (784, 16, 16, 10)
train_data_chunk_size = 50

metrics_queue_size = 3
metrics_queue_batch_size = 5


def matrix_to_xv(m):
    return np.array([d for r in m for d in r]) / 255


def digit_to_yv(d):
    if not 0 <= d <= 9:
        raise ValueError('Expected digit between 0 and 9')
    y = np.zeros(shape=10)
    y[d] = 1
    return y


def get_avg_components(list_of_vectors):
    return [statistics.mean(e) for e in zip(*list_of_vectors)]


def train(queue, queue_batch_size, ai_instance: ai.Ai, xs, ys, c_size, c_count=None):
    x_chunks = iu.get_array_chunks(xs, c_size, c_count, return_incomplete=False)
    y_chunks = iu.get_array_chunks(ys, c_size, c_count, return_incomplete=False)
    xy_chunks = zu.zip2(x_chunks, y_chunks)
    xy_chunks_chunks = iu.get_chunks(xy_chunks, queue_batch_size)
    for xy_chunks_chunk in xy_chunks_chunks:
        with TimeLog('TRAIN'):
            metrics_batch = []
            for x_chunk, y_chunk in xy_chunks_chunk:
                fxc = [matrix_to_xv(x) for x in x_chunk]
                fya = [digit_to_yv(y) for y in y_chunk]
                metric = ai_instance.train(fxc, fya)
                metrics_batch.append(metric)
            queue.put_nowait(metrics_batch)
    queue.put_nowait(None)
    print('END TRAIN')


def create_app():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = pg.mkQApp("AI trainer")
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # set stylesheet
    file = QFile(":dark-theme")
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())

    return app


def main():
    queue = mp.Queue(maxsize=metrics_queue_size)

    app = create_app()
    window = MainWindow(queue, train_data_chunk_size)
    window.show()

    ai_model = ai.init_model(layer_sizes)
    train_args = (queue, metrics_queue_batch_size, ai_model, x_train, y_train, train_data_chunk_size)
    train_process = mp.Process(target=train, args=train_args, daemon=True)
    train_process.start()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
