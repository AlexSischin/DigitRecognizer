import multiprocessing as mp
import statistics
import sys

import numpy as np
import pyqtgraph as pg
import tensorflow as tf
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

import ai
import qrc_resources
from ui.main_window import MainWindow
from utils.zip_utils import zip2

# To save from imports optimization by IDEs
qrc_resources = qrc_resources

train_data, test_data = tf.keras.datasets.mnist.load_data()
layer_sizes = (784, 16, 16, 10)
careful_learn_threshold = .1
train_data_chunk_size = 50
careful_train_data_chunk_size = 250

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


def train(queue, queue_batch_size, ai_instance: ai.Ai, data, cl_threshold, c_size, cl_c_size=None):
    x_data, y_data = data
    last_costs = []
    last_costs_size = 10
    last_costs_cl_threshold = 5
    metrics_batch = []
    xy_chunk = []
    careful_train = False
    for x, y in zip2(x_data, y_data):
        x_vector = matrix_to_xv(x)
        y_vector = digit_to_yv(y)
        xy_chunk.append((x_vector, y_vector))

        cur_chunk_size = cl_c_size if careful_train else c_size
        if len(xy_chunk) >= cur_chunk_size:
            x_vectors, y_vectors = zip(*xy_chunk)
            xy_chunk.clear()
            patch_gradient = True or not careful_train
            metric = ai_instance.train(x_vectors, y_vectors, patch_gradient)
            metrics_batch.append(metric)

            last_costs.append(metric.cost)
            if len(last_costs) >= last_costs_size:
                last_costs.pop(0)
            careful_train = len([1 for c in last_costs if c < cl_threshold]) >= last_costs_cl_threshold

            if len(metrics_batch) >= queue_batch_size:
                queue.put_nowait(metrics_batch)
                metrics_batch = []
    if metrics_batch:
        queue.put_nowait(metrics_batch)
    queue.put_nowait(None)
    print('END TRAIN')


def create_app():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = pg.mkQApp("AI trainer")
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    return app


def main():
    queue = mp.Queue(maxsize=metrics_queue_size)

    app = create_app()
    window = MainWindow(queue, test_data)
    window.show()

    ai_model = ai.Ai(layer_sizes)
    train_args = (queue, metrics_queue_batch_size, ai_model, train_data,
                  careful_learn_threshold, train_data_chunk_size, careful_train_data_chunk_size)
    train_process = mp.Process(target=train, args=train_args, daemon=True)
    train_process.start()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
