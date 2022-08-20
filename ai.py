import numpy as np
import math


def scalar_sigma(x):
    return 1 / (1 + pow(math.e, -x))


def sigma(v: np.ndarray) -> np.ndarray:
    return np.array([scalar_sigma(x) for x in v])


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
        if len(bias) < 1:
            raise ValueError(f'Bias must be not empty (depth={depth})')
        if weight.shape[1] != bias.shape[0]:
            raise ValueError(f'Weight columns count must be equal to bias rows count (depth={depth})')


class Ai:
    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        validate_brain(weights, biases)
        self.w = np.array(weights)
        self.b = np.array(biases)
        self.depth = len(weights)

    def feed(self, v: np.ndarray) -> list[np.ndarray]:
        activations = [v]
        for d in range(self.depth):
            v = sigma(v.dot(self.w[d]) + self.b[d])
            activations.append(v)
        return activations

    def nudge(self):
        pass


def create_random(layer_sizes) -> Ai:
    if len(layer_sizes) < 2:
        raise ValueError('Expected 2 or more layers')
    biases = []
    weights = []
    prev_layer_size = layer_sizes[0]
    for layer_size in layer_sizes[1:]:
        biases += [np.random.random_sample(size=layer_size) * 10 - 5]
        weights += [np.random.random_sample(size=(layer_size, prev_layer_size)) * 10 - 5]
    return Ai(weights, biases)
