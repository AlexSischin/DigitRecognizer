import math
from dataclasses import dataclass

import numpy as np

from utils import zip_utils as zp
from utils.iter_utils import is_iterable


class ActivationFunction:

    @staticmethod
    def of_vec(xv: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def der_of(x: float) -> float:
        pass


class SigmoidFunc(ActivationFunction):

    @staticmethod
    def _of(x: float) -> float:
        return 1 / (1 + pow(math.e, -x))

    @staticmethod
    def of_vec(xv: np.ndarray) -> np.ndarray:
        return np.vectorize(SigmoidFunc._of)(xv)

    @staticmethod
    def der_of(x: float) -> float:
        s = SigmoidFunc._of(x)
        return s * (1 - s)


class ReLuFunc(ActivationFunction):

    @staticmethod
    def _of(x: float) -> float:
        return max(0., x)

    @staticmethod
    def of_vec(xv: np.ndarray) -> np.ndarray:
        return np.vectorize(ReLuFunc._of)(xv)

    @staticmethod
    def der_of(x: float) -> float:
        return 0 if x < 0 else 1


def validate_brain(weights: list[np.ndarray], biases: list[np.ndarray]):
    depth = len(weights)
    if len(biases) != depth:
        raise ValueError('Weights and biases must have the same depth')
    if depth < 1:
        raise ValueError('Depth must be greater than 0')
    for weight, bias in list(zip(weights, biases)):
        if len(weight.shape) != 2:
            raise ValueError(f'Weight must be two-dimensional (depth={depth})')
        if 0 in weight.shape:
            raise ValueError(f'Weight must be not empty (depth={depth})')
        if len(bias.shape) != 1:
            raise ValueError(f'Bias must be one-dimensional (depth={depth})')
        if 0 in bias.shape:
            raise ValueError(f'Bias must be not empty (depth={depth})')
        if weight.shape[0] != bias.shape[0]:
            raise ValueError(f'Weight columns count must be equal to bias length (depth={depth})')


def calc_z_factor_derivatives(n: int, a_der: np.ndarray, z_factor: np.ndarray, act_func_der):
    z_der = np.empty(shape=n)
    for j in range(n):
        z_der[j] = act_func_der(z_factor[j]) * a_der[j]
    return z_der


def calc_weight_derivatives(n: int, n_next: int, z_der: np.ndarray, a_next: np.ndarray):
    w_der = np.empty(shape=(n, n_next))
    for k in range(n_next):
        for j in range(n):
            w_der[j, k] = a_next[k] * z_der[j]
    return w_der


def calc_root_activation_derivatives(actual_result: np.ndarray, expected_result: np.ndarray):
    return np.array([2 * (a - y) for a, y in zp.zip2(actual_result, expected_result)])


def calc_next_activation_derivatives(n: int, n_next: int, z_der: np.ndarray, weight: np.ndarray):
    a_der_next = np.empty(shape=n_next)
    for k in range(n_next):
        a_der_next[k] = 0
        for j in range(n):
            a_der_next[k] += weight[j, k] * z_der[j]
    return a_der_next


@dataclass(frozen=True)
class TrainMetric:
    data_used: int
    w: list[np.ndarray]
    b: list[np.ndarray]
    w_gradient: list[np.ndarray]
    b_gradient: list[np.ndarray]
    gradient_len: float
    costs: list[np.ndarray]
    cost: float
    inputs: list[np.ndarray]
    outputs: list[np.ndarray]
    expected: list[np.ndarray]


def square_mean_costs(costs: list[np.ndarray]) -> np.ndarray:
    costs_array = np.array(costs)
    squared_costs = np.square(costs_array)
    return np.mean(np.sum(squared_costs, axis=1))


def generate_weights_and_biases(layer_sizes):
    if len(layer_sizes) < 2:
        raise ValueError('Expected 2 or more layers')
    biases = []
    weights = []
    for layer_size, prev_layer_size in zip(layer_sizes[1:], layer_sizes):
        biases += [np.random.random_sample(size=layer_size) * 1 - 0.5]
        weights += [np.random.random_sample(size=(layer_size, prev_layer_size)) * 1 - 0.5]
    return weights, biases


class Ai:
    def __init__(self, layer_sizes=None, weights=None, biases=None, activation_functions=None) -> None:
        if layer_sizes:
            weights, biases = generate_weights_and_biases(layer_sizes)
        if not activation_functions:
            activation_functions = [SigmoidFunc() for _ in biases]
        elif not is_iterable(activation_functions):
            activation_functions = [activation_functions for _ in biases]

        validate_brain(weights, biases)
        self.w: list[np.ndarray] = weights
        self.b: list[np.ndarray] = biases
        self.activation_functions: list[ActivationFunction] = activation_functions
        self._data_used = 0

    def feed(self, x: np.ndarray) -> np.ndarray:
        return self._feed(x)[1][-1]

    def train(self,
              x_vectors: list[np.ndarray],
              y_vectors: list[np.ndarray],
              learning_rate: float = None
              ) -> TrainMetric:
        w_gradient_sum = [np.zeros(w.shape) for w in self.w]
        b_gradient_sum = [np.zeros(b.shape) for b in self.b]
        costs = []
        outputs = []
        for x_vec, y_vec in zp.zip2(x_vectors, y_vectors):
            z_state, a_state = self._feed(x_vec)
            w_local_gradient, b_local_gradient = self._backpropagate(z_state, a_state, y_vec)
            w_gradient_sum = [ws + w for ws, w in zp.zip2(w_gradient_sum, w_local_gradient)]
            b_gradient_sum = [bs + b for bs, b in zp.zip2(b_gradient_sum, b_local_gradient)]
            costs.append(np.array([(y - a) ** 2 for y, a in zp.zip2(y_vec, a_state[-1])]))
            outputs.append(a_state[-1])
        n = len(x_vectors)
        w_gradient = [w / n for w in w_gradient_sum]
        b_gradient = [b / n for b in b_gradient_sum]
        cost = square_mean_costs(costs)
        gradient_len = math.sqrt(sum([np.sum(gc ** 2) for gc in w_gradient + b_gradient]))
        if gradient_len != 0:
            if learning_rate is None:
                learning_rate = 1 / gradient_len
            self._patch(w_gradient, b_gradient, learning_rate)
        self._data_used += len(x_vectors)
        return TrainMetric(
            data_used=self._data_used, w=self.w, b=self.b, w_gradient=w_gradient, b_gradient=b_gradient,
            gradient_len=gradient_len, costs=costs, cost=cost, inputs=x_vectors, outputs=outputs, expected=y_vectors
        )

    def _patch(self, w_gradient: list[np.ndarray], b_gradient: list[np.ndarray], learning_rate):
        self.w = [w - wd * learning_rate for w, wd in zp.zip2(self.w, w_gradient)]
        self.b = [b - bd * learning_rate for b, bd in zp.zip2(self.b, b_gradient)]

    def _feed(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if x.ndim > 1:
            x = x.flatten()
        if not all(0 <= xi <= 1 for xi in x):
            raise ValueError("All elements must have value in range: [0, 1]")
        z_factors = []
        activations = [x]
        for w, b, act_func in zp.zip3(self.w, self.b, self.activation_functions):
            z = np.dot(w, x) + b
            x = act_func.of_vec(z)
            z_factors.append(z)
            activations.append(x)
        return z_factors, activations

    def _backpropagate(self, z_factors: list[np.ndarray], activations: list[np.ndarray], expected_result: np.ndarray):
        w_derivatives, b_derivatives = [], []
        a_der = calc_root_activation_derivatives(activations[-1], expected_result)
        for l_weights, l_act_func, l_z_factors, next_l_activations in zp.zip4(
                self.w[::-1], self.activation_functions[::-1], z_factors[::-1], activations[-2::-1]):
            n, n_next = l_weights.shape

            b_der = z_der = calc_z_factor_derivatives(n, a_der, l_z_factors, l_act_func.der_of)
            w_der = calc_weight_derivatives(n, n_next, z_der, next_l_activations)

            w_derivatives.insert(0, w_der)
            b_derivatives.insert(0, b_der)

            a_der = calc_next_activation_derivatives(n, n_next, z_der, l_weights)

        return w_derivatives, b_derivatives
