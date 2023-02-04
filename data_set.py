import numpy as np
import tensorflow as tf


# [0, 255] -> [0, 1]
def scale(x: np.ndarray):
    return x / 255


# [0, 1] -> [0, 255]
def unscale(x: np.ndarray):
    return x * 255


# 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def vectorize(y: np.ndarray):
    y_vec = np.zeros((y.size, 10), dtype=float)
    y_vec[np.arange(len(y)), y] = 1
    return y_vec


# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] -> 3
def unvectorize(y: np.ndarray):
    if y.ndim == 1:
        return y.argmax()
    elif y.ndim == 2:
        return y.argmax(axis=0)
    else:
        raise ValueError('Y must be 1D or 2D numpy array')


(raw_train_x, raw_train_y), (raw_test_x, raw_test_y) = tf.keras.datasets.mnist.load_data()
train_x, test_x = scale(raw_train_x), scale(raw_test_x)
train_y, test_y = vectorize(raw_train_y), vectorize(raw_test_y)
