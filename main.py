import dataclasses
import random
import sys
from configparser import ConfigParser

import numpy as np
import tensorflow as tf
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

import ai
import resources.qrc as qrc_resources
from trainer import AiTrainer, random_extended_chunked_list
from utils import cfg_parsing
from utils.iter_utils import is_iterable
from utils.zip_utils import zip2

# To save from imports optimization by IDEs
qrc_resources = qrc_resources


@dataclasses.dataclass(frozen=True)
class AiArgs:
    layers: tuple[int, ...]
    activation_functions: tuple[ai.ActivationFunction, ...]


@dataclasses.dataclass(frozen=True)
class TrainArgs:
    chunk_size: int
    chunk_count: int


@dataclasses.dataclass(frozen=True)
class ProcessingArgs:
    queue_max_size: int
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


def load_train_and_test_data(chunk_size, chunk_count):
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    train_x = normalize_image(train_x)
    train_y = digit_to_yv_vec(train_y)
    test_x = normalize_image(test_x)
    test_y = digit_to_yv_vec(test_y)

    train_data_set = list(zip2(train_x, train_y))
    test_data = list(zip2(test_x, test_y))
    train_data = random_extended_chunked_list(train_data_set, chunk_size, chunk_count)
    random.shuffle(test_data)

    return train_data, test_data


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
        layers=ai_section.get_int_tuple('layers'),
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
    return ProcessingArgs(
        queue_max_size=processing_section.getint('queue max size'),
        queue_batch_size=processing_section.getint('queue batch size')
    )


def main():
    cfg = create_cfg()
    cfg.read('resources/app.ini')
    ai_args = get_ai_args(cfg)
    train_args = get_train_args(cfg)
    processing_args = get_processing_args(cfg)
    train_data, test_data = load_train_and_test_data(train_args.chunk_size, train_args.chunk_count)

    ai_model = ai.Ai(layer_sizes=ai_args.layers, activation_functions=ai_args.activation_functions)

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    trainer_app = AiTrainer(
        ai_model=ai_model,
        train_data=train_data,
        test_data=test_data,
        queue_max_size=processing_args.queue_max_size,
        queue_batch_size=processing_args.queue_batch_size
    )
    trainer_app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    sys.exit(trainer_app.exec())


if __name__ == '__main__':
    main()
