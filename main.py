import multiprocessing
import statistics

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import ai
from animation.trainfig import TrainFig
from utils import iter_utils as iu
from utils import zip_utils as zu
from utils.time_utils import TimeLog

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
layer_sizes = (784, 16, 16, 10)
chunk_size = 50

queue_size = 2
metrics_batch_size = 5
train_animation_interval = 1000
metrics_range = 50


def matrix_to_xv(m):
    return np.array([d for r in m for d in r]) / 255


def digit_to_yv(d):
    if not 0 <= d <= 9:
        raise ValueError('Expected digit between 0 and 9')
    y = np.zeros(shape=10)
    y[d] = 1
    return y


def get_avg_components(list_of_vectors):
    return [statistics.mean(e) for e in zip(*list_of_vectors)]


def train(queue, batch_size, ai_instance: ai.Ai, xs, ys, c_size, c_count=None):
    x_chunks = iu.get_array_chunks(xs, c_size, c_count)
    y_chunks = iu.get_array_chunks(ys, c_size, c_count)
    xy_chunks = zu.zip2(x_chunks, y_chunks)
    xy_chunks_chunks = iu.get_chunks(xy_chunks, batch_size)
    for xy_chunks_chunk in xy_chunks_chunks:
        with TimeLog('TRAIN'):
            metrics_buff = []
            for x_chunk, y_chunk in xy_chunks_chunk:
                fxc = [matrix_to_xv(x) for x in x_chunk]
                fya = [digit_to_yv(y) for y in y_chunk]
                metric = ai_instance.train(fxc, fya)
                metrics_buff.append(metric)
            queue.put_nowait(metrics_buff.copy())
    print('END TRAIN')


def main():
    ai_model = ai.init_model(layer_sizes)

    queue = multiprocessing.Queue(maxsize=queue_size)
    train_args = (queue, metrics_batch_size, ai_model, x_train, y_train, chunk_size)
    train_process = multiprocessing.Process(None, train, args=train_args, daemon=True)
    train_process.start()

    train_fig = TrainFig(queue, animation_interval=train_animation_interval, metrics_range=metrics_range)
    _ta = train_fig.animation

    plt.tight_layout()
    plt.show()
    plt.close('all')

    train_fig.fig.savefig('charts/temp.png')
    # test(ai_model, x_test, y_test)


if __name__ == '__main__':
    main()
