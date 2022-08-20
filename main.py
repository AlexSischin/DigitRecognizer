import random

import numpy as np

import ai


layer_sizes = (4, 4)


def get_random_array(size):
    return np.array([random.randint(0, 1) for v in range(size)])


def sample_func(input_vector: np.ndarray):
    output = []
    for i in input_vector:
        output += [1 - i]
    return np.array(output)


def get_cost_vector(expected_result, actual_result):
    return [(e - a) * (e - a) for e, a in list(zip(expected_result, actual_result))]


def get_cost(ai_instance: ai.Ai, iterations):
    cost = 0
    for iteration in range(iterations):
        input_vector = get_random_array(layer_sizes[0])
        expected_result = sample_func(input_vector)
        activations = ai_instance.feed(input_vector)
        actual_result = activations[-1]
        nudge_vector = expected_result - actual_result
        cost_vector = get_cost_vector(expected_result, actual_result)
        cost_sum = sum(cost_vector)
        print(f'input_vector = \t\t{input_vector}')
        print(f'expected_result = \t{expected_result}')
        print(f'actual_result = \t{["%0.2f" % i for i in actual_result]}')
        print(f'cost_vector = \t\t{["%0.2f" % i for i in cost_vector]}')
        print(f'cost_sum = \t\t\t{cost_sum}\n')
        cost += cost_sum
    return cost / iterations


def main():
    ai_instance = ai.create_random(layer_sizes)
    cost = get_cost(ai_instance, 5)
    print(f'Cost:\t{cost}')


if __name__ == '__main__':
    main()
