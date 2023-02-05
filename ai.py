from dataclasses import dataclass
from numbers import Number
from typing import Callable

import numpy as np

from utils import zip_utils as zp


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
        return 1 / (1 + np.exp(-x))

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


def validate_brain(weights: tuple[np.ndarray], biases: tuple[np.ndarray]):
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


@dataclass(frozen=True)
class TrainMetric:
    data_used: int
    w: tuple[np.ndarray, ...]
    b: tuple[np.ndarray, ...]
    w_gradient: tuple[np.ndarray, ...]
    b_gradient: tuple[np.ndarray, ...]
    gradient_len: float
    costs: tuple[np.ndarray, ...]
    cost: float
    inputs: tuple[np.ndarray, ...]
    outputs: tuple[np.ndarray, ...]
    expected: tuple[np.ndarray, ...]


def square_mean_costs(costs: list[np.ndarray]) -> np.ndarray:
    costs_array = np.array(costs)
    squared_costs = np.square(costs_array)
    return np.mean(np.sum(squared_costs, axis=1))


def default_learning_rate(gradient_length, **_):
    return 1 / gradient_length if gradient_length != 0 else 0


class Ai:
    def __init__(self,
                 weights: tuple[np.ndarray, ...],
                 biases: tuple[np.ndarray, ...],
                 activation_functions: tuple[ActivationFunction, ...] = None,
                 learning_rate: Number | Callable = None
                 ) -> None:
        if activation_functions is None:
            activation_functions = [SigmoidFunc() for _ in biases]
        if learning_rate is None:
            learning_rate = default_learning_rate

        validate_brain(weights, biases)
        self.w: tuple[np.ndarray] = weights
        self.b: tuple[np.ndarray] = biases
        self.f: tuple[ActivationFunction] = activation_functions
        self.learning_rate = learning_rate
        self.data_used = 0

    def feed(self, x: np.ndarray) -> np.ndarray:
        return self._feed(x)[1][-1]

    def train(self,
              x_vectors: list[np.ndarray],
              y_vectors: list[np.ndarray]
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
        gradient_len = np.sqrt(sum([np.sum(gc ** 2) for gc in w_gradient + b_gradient]))
        if gradient_len != 0:
            learning_rate = self._get_learning_rate(cost, gradient_len)
            self._patch(w_gradient, b_gradient, learning_rate)
        self.data_used += len(x_vectors)
        return TrainMetric(
            data_used=self.data_used, w=self.w, b=self.b, w_gradient=tuple(w_gradient), b_gradient=tuple(b_gradient),
            gradient_len=gradient_len, costs=tuple(costs), cost=cost, inputs=tuple(x_vectors),
            outputs=tuple(outputs), expected=tuple(y_vectors)
        )

    def _patch(self, w_gradient: list[np.ndarray], b_gradient: list[np.ndarray], learning_rate):
        self.w = [w - wd * learning_rate for w, wd in zp.zip2(self.w, w_gradient)]
        self.b = [b - bd * learning_rate for b, bd in zp.zip2(self.b, b_gradient)]

    def _get_learning_rate(self, cost, gradient_length):
        if isinstance(self.learning_rate, Number):
            return self.learning_rate
        if isinstance(self.learning_rate, Callable):
            return self.learning_rate(cost=cost, gradient_length=gradient_length)
        raise ValueError('Learn rate must be Number or Callable')

    def _feed(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if x.ndim > 1:
            x = x.flatten()
        if not all(0 <= xi <= 1 for xi in x):
            raise ValueError("All elements must have value in range: [0, 1]")
        z_factors = []
        activations = [x]
        for w, b, f in zp.zip3(self.w, self.b, self.f):
            z = np.dot(w, x) + b
            x = f.of_vec(z)
            z_factors.append(z)
            activations.append(x)
        return z_factors, activations

    def _backpropagate(self, z: list[np.ndarray], a: list[np.ndarray], y: np.ndarray):
        dj_dw_list, dj_db_list = [], []
        dj_da = (a[-1] - y) * 2
        for w_l, f, z_l, a_pl in zp.zip4(self.w[::-1], self.f[::-1], z[::-1], a[-2::-1]):
            dj_db = dj_dz = f.der_of(z_l) * dj_da
            dj_dw = np.expand_dims(dj_dz, 1) @ np.expand_dims(a_pl, 0)
            dj_da = w_l.T @ dj_dz
            dj_dw_list.insert(0, dj_dw)
            dj_db_list.insert(0, dj_db)

        return dj_dw_list, dj_db_list
