import numpy as np
import tensorflow as tf
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from data_set import train_x
from distribution_viewer.distributions import LayerParams, calc_distributions
from distribution_viewer.distributions_app import DistributionViewerApp
from functions import FormulaFunction, get_relu_inv_approx_func, sigmoid_inv, sigmoid_der, get_relu_der_approx_func, \
    get_uniform_pdf, \
    get_gaussian_pdf
from resources import app_ini
from resources.app_ini import ActivationFunction, Distribution, DistributionType, DistributionParam, AiCfg
from utils.time_utils import TimeLog
from utils.zip_utils import zip4


def get_layer_params(cfg: AiCfg):
    layer_params = []
    for prev_layer_size, act_func, w_distribution, b_distribution in zip4(
            cfg.layers[:-1], cfg.activation_functions, cfg.weight_distributions, cfg.bias_distributions):
        f_inv_a = get_activation_function_inv(act_func)
        f_der_a = get_activation_function_der(act_func)
        f_w = get_distribution_function(w_distribution)
        f_b = get_distribution_function(b_distribution)
        params = LayerParams(prev_layer_size, f_inv_a, f_der_a, f_w, f_b)
        layer_params.append(params)
    return tuple(layer_params)


def get_activation_function_inv(func: ActivationFunction, relu_approx_factor=3) -> FormulaFunction:
    if func == ActivationFunction.SIGMOID:
        return FormulaFunction(sigmoid_inv, domain=(0.00001, 0.99999))
    elif func == ActivationFunction.RELU:
        return FormulaFunction(get_relu_inv_approx_func(relu_approx_factor), domain=(0.00001, +np.inf))
    raise ValueError(f'Invalid activation function: {func}')


def get_activation_function_der(func: ActivationFunction, relu_approx_factor=3) -> FormulaFunction:
    if func == ActivationFunction.SIGMOID:
        return FormulaFunction(sigmoid_der)
    elif func == ActivationFunction.RELU:
        return FormulaFunction(get_relu_der_approx_func(relu_approx_factor))
    raise ValueError(f'Invalid activation function: {func}')


def get_distribution_function(func: Distribution) -> FormulaFunction:
    f_type = func.type
    if f_type == DistributionType.UNIFORM:
        a = func.params[DistributionParam.LB.value]
        b = func.params[DistributionParam.RB.value]
        return FormulaFunction(get_uniform_pdf(a, b))
    elif f_type == DistributionType.GAUSSIAN:
        m = func.params[DistributionParam.MEAN.value]
        sd = func.params[DistributionParam.SD.value]
        return FormulaFunction(get_gaussian_pdf(m, sd))
    raise ValueError(f'Invalid distribution: {func}')


def run_distribution_viewer():
    cfg = app_ini.cfg.ai
    domain = np.linspace(-10, +10, 1000)
    train_data = train_x.flatten()
    layer_params = get_layer_params(cfg)
    with TimeLog('Distributions calculation') as _:
        layer_distributions = calc_distributions(domain, train_data, layer_params)

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = DistributionViewerApp(cfg.layers, layer_distributions, domain)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app.exec()


if __name__ == '__main__':
    run_distribution_viewer()
