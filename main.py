import sys

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

import ai
import resources.qrc as qrc_resources
from data_set import train_x, train_y, test_x, test_y
from resources import app_ini
from resources.app_ini import AiCfg, Distribution, DistributionType, DistributionParam
from trainer import AiTrainer
from utils.iter_utils import random_extended_chunked_list
from utils.zip_utils import zip2, zip3

# To save from imports optimization by IDEs
qrc_resources = qrc_resources

activation_funcs = {
    app_ini.ActivationFunction.SIGMOID: ai.SigmoidFunc(),
    app_ini.ActivationFunction.RELU: ai.ReLuFunc()
}


# Using Callable class because it's impossible to pickle local objects (decorator funcs)
class LearningRate:
    def __init__(self, learning_rate_map) -> None:
        super().__init__()
        self.learning_rate_map = learning_rate_map

    def __call__(self, cost, gradient_length, **_):
        if gradient_length == 0:
            return 0
        key = max([c for c in self.learning_rate_map.keys() if c <= cost])
        val = self.learning_rate_map[key]
        if val <= 0:
            return 1 / gradient_length
        else:
            return val


def prepare_data(chunk_size, chunk_count):
    train_data = list(zip2(train_x, train_y))
    test_data = list(zip2(test_x, test_y))
    train_data = random_extended_chunked_list(train_data, chunk_size, chunk_count)
    np.random.shuffle(test_data)
    return train_data, test_data


def generate_numbers(shape: int | tuple[int, ...], distribution: Distribution):
    d_type = distribution.type
    if d_type == DistributionType.UNIFORM:
        a = distribution.params[DistributionParam.LB.value]
        b = distribution.params[DistributionParam.RB.value]
        return np.random.uniform(low=a, high=b, size=shape)
    elif d_type == DistributionType.GAUSSIAN:
        m = distribution.params[DistributionParam.MEAN.value]
        sd = distribution.params[DistributionParam.SD.value]
        return np.random.normal(loc=m, scale=sd, size=shape)
    raise ValueError(f'Invalid distribution: {distribution}')


def generate_weights_and_biases(cfg: AiCfg):
    layers = cfg.layers
    w_distributions = cfg.weight_distributions
    b_distributions = cfg.bias_distributions
    if len(layers) < 2:
        raise ValueError('Expected 2 or more layers')

    weights = []
    for layer_size, prev_layer_size, w_distr in zip3(layers[1:], layers[:-1], w_distributions):
        w = generate_numbers(shape=(layer_size, prev_layer_size), distribution=w_distr)
        weights.append(w)

    biases = []
    for layer_size, b_distr in zip2(layers[1:], b_distributions):
        b = generate_numbers(shape=layer_size, distribution=b_distr)
        biases.append(b)

    return tuple(weights), tuple(biases)


def main():
    cfg = app_ini.cfg

    learning_rate = LearningRate(cfg.ai.learning_rate)
    train_data, test_data = prepare_data(cfg.train.chunk_size, cfg.train.chunk_count)

    activation_functions = tuple([activation_funcs[f] for f in cfg.ai.activation_functions])

    w, b = generate_weights_and_biases(cfg.ai)
    # TODO pass learning_rate to trainer, not AI
    ai_model = ai.Ai(w, b, activation_functions=activation_functions, learning_rate=learning_rate)

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    trainer_app = AiTrainer(
        ai_model=ai_model,
        train_data=train_data,
        test_data=test_data,
        queue_max_size=cfg.processing.queue_max_size,
        queue_batch_size=cfg.processing.queue_batch_size
    )
    trainer_app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    sys.exit(trainer_app.exec())


if __name__ == '__main__':
    main()
