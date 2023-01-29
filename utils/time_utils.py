import functools
from random import randint
from timeit import default_timer as timer


class TimeLog:
    def __init__(self, label='x') -> None:
        super().__init__()
        self.label = label
        self.rid = randint(0, 1000000)

    def __enter__(self):
        self.begin_time = timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_s = timer() - self.begin_time
        print(f'[{self.rid}] Time elapsed on {self.label} is: {elapsed_s}s')


def time_log(label='x'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with TimeLog(label):
                return func(*args, **kwargs)

        return wrapper

    return decorator
