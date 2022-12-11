import dataclasses
import multiprocessing as mp
import random
import sys
from configparser import ConfigParser

import numpy as np
import pyqtgraph as pg
import tensorflow as tf
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

import ai
from utils import cfg_parsing
import resources.qrc as qrc_resources
from ui.main_window import MainWindow
from utils.iter_utils import is_iterable
from utils.zip_utils import zip2

# To save from imports optimization by IDEs
qrc_resources = qrc_resources


@dataclasses.dataclass(frozen=True)
class AiArgs:
    layer_sizes: tuple[int, ...]
    activation_functions: tuple[ai.ActivationFunction, ...]


@dataclasses.dataclass(frozen=True)
class TrainArgs:
    chunk_size: int
    chunk_count: int


@dataclasses.dataclass(frozen=True)
class ProcessArgs:
    queue_size: int
    queue_batch_size: int


def digit_to_yv(d):
    if not 0 <= d <= 9:
        raise ValueError('Expected digit between 0 and 9')
    y = np.zeros(shape=10)
    y[d] = 1
    return y


def digit_to_yv_vec(dm):
    return np.array([
        digit_to_yv_vec(d) if is_iterable(d) else digit_to_yv(d)
        for d in dm
    ])


def normalize_image(img: np.ndarray):
    return img / 255


def load_train_and_test_data():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    train_x = normalize_image(train_x)
    train_y = digit_to_yv_vec(train_y)
    test_x = normalize_image(test_x)
    test_y = digit_to_yv_vec(test_y)

    train_data = list(zip2(train_x, train_y))
    test_data = list(zip2(test_x, test_y))
    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data


def train(queue, queue_batch_size, ai_model: ai.Ai, train_data, chunk_size):
    metrics_batch = []
    xy_chunk = []
    for x, y in train_data:
        xy_chunk.append((x, y))

        if len(xy_chunk) >= chunk_size:
            xs, ys = zip(*xy_chunk)
            xy_chunk.clear()
            metric = ai_model.train(xs, ys)
            metrics_batch.append(metric)

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


def create_cfg():
    converters = {
        '_int_tuple': cfg_parsing.to_tuple(int),
        '_act_func_tuple': cfg_parsing.to_tuple_f_dict({'Sigmoid': ai.SigmoidFunc,
                                                       'ReLU': ai.ReLuFunc})
    }
    return ConfigParser(converters=converters)


def get_ai_args(cfg: ConfigParser):
    ai_section = cfg['AI']
    return AiArgs(
        layer_sizes=ai_section.get_int_tuple('layer sizes'),
        activation_functions=ai_section.get_act_func_tuple('activation functions'),
    )


def get_train_args(cfg: ConfigParser):
    train_section = cfg['Train']
    return TrainArgs(
        chunk_size=train_section.getint('chunk size'),
        chunk_count=train_section.getint('chunk count')
    )


def get_processing_args(cfg: ConfigParser):
    processing_section = cfg['Processing']
    return ProcessArgs(
        queue_size=processing_section.getint('queue size'),
        queue_batch_size=processing_section.getint('queue batch size')
    )


def main():
    cfg = create_cfg()
    cfg.read('resources/app.ini')
    ai_args = get_ai_args(cfg)
    train_args = get_train_args(cfg)
    processing_args = get_processing_args(cfg)

    train_data, test_data = load_train_and_test_data()
    queue = mp.Queue(maxsize=processing_args.queue_size)
    layer_count = len(ai_args.layer_sizes)

    app = create_app()

    window = MainWindow(queue, test_data, layer_count, ai_args.activation_functions)
    window.show()

    ai_model = ai.Ai(layer_sizes=ai_args.layer_sizes, activation_functions=ai_args.activation_functions)
    train_process_args = (queue, processing_args.queue_batch_size, ai_model, train_data, train_args.chunk_size)
    train_process = mp.Process(target=train, args=train_process_args, daemon=True)
    train_process.start()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
