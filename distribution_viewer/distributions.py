import dataclasses
from functools import reduce

import numpy as np

from functions import Function, product_distribution, sum_distribution, transform_distribution, \
    ApproximateFunction


@dataclasses.dataclass(frozen=True)
class LayerParams:
    prev_layer_size: int
    activation_function_inv: Function
    activation_function_der: Function
    weights_pdf: Function
    biases_pdf: Function


@dataclasses.dataclass(frozen=True)
class LayerDistributions:
    W: Function  # Weights
    B: Function  # Biases
    A_prev: Function  # Activations in the left layer
    P: Function  # Weighted activations in the input layer
    S: Function  # Sum of weighted activations in the input layer
    Z: Function  # Sum of weighted activations in the input layer with bias
    A: Function  # Activations in the right layer


def calc_pdf(domain, data: np.ndarray) -> ApproximateFunction:
    dx = domain[1] - domain[0]
    data_array = np.array(data)
    bins = np.append(domain, domain[-1] + dx) - dx / 2
    hist, _ = np.histogram(data_array, bins)
    area = np.trapz(hist, dx=dx)
    f_x = hist / area
    return ApproximateFunction(domain, f_x)


def calc_distributions(domain, input_data, layer_params_list: tuple[LayerParams]) -> tuple[LayerDistributions]:
    layer_distributions = []
    f_a_prev = calc_pdf(domain, input_data)
    for layer in layer_params_list:
        f_w = layer.weights_pdf
        f_b = layer.biases_pdf
        f_p = product_distribution(domain, f_a_prev, f_w)
        f_s = reduce(lambda a, b: sum_distribution(domain, a, b), [f_p for _ in range(layer.prev_layer_size)])
        f_z = sum_distribution(domain, f_s, f_b)
        f_a = transform_distribution(domain, f_z, layer.activation_function_inv, layer.activation_function_der)
        distributions = LayerDistributions(f_w, f_b, f_a_prev, f_p, f_s, f_z, f_a)
        layer_distributions.append(distributions)
        f_a_prev = f_a
    return tuple(layer_distributions)
