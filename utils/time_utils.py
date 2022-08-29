import functools
from timeit import default_timer as timer
from random import randint


class TimeLog:
    def __init__(self, label='x', log_start=False) -> None:
        super().__init__()
        self.label = label
        self.log_start = log_start
        self.rid = randint(0, 1000000)

    def __enter__(self):
        if self.log_start:
            print('[{:0>6}] () - {} - ()'.format(self.rid, self.label))
        self.begin_time = timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = timer()
        begin_time_mcs = self.begin_time * 10 ** 6
        end_time_mcs = end_time * 10 ** 6
        elapsed = end_time_mcs - begin_time_mcs
        print('[{:0>6}] {:.0f} - {} - {:.0f} ({:.0f} μs)'
              .format(self.rid, begin_time_mcs, self.label, end_time_mcs, elapsed))


def time_log(label='x', log_start=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with TimeLog(label, log_start):
                return func(*args, **kwargs)

        return wrapper

    return decorator
