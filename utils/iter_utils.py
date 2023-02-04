import random
from typing import Generator, TypeVar

T = TypeVar('T')


class IncompleteLastChunk(Exception):
    pass


class IterableExceed(Exception):
    pass


def limited_iterator(iterable, limit: int):
    while limit:
        yield next(iterable)
        limit -= 1


def get_array_chunks(ndarray, size, count=None, return_incomplete=True):
    max_pos = len(ndarray) if return_incomplete else len(ndarray) // size * size
    chunks = (
        ndarray[pos:min(pos + size, max_pos)]
        for pos in range(0, len(ndarray), size)
    )
    if count:
        yield from limited_iterator(chunks, count)
    else:
        yield from chunks


def get_chunks(iterable, size, count=None, return_incomplete=True):
    def generator():
        buff = []
        while True:
            try:
                v = next(iterable)
            except StopIteration:
                if buff and return_incomplete:
                    yield buff
                return
            buff.append(v)
            if len(buff) >= size:
                yield buff
                buff = []

    if count:
        yield from limited_iterator(generator(), count)
    else:
        yield from generator()


def is_iterable(val):
    try:
        iter(val)
    except TypeError:
        return False
    return True


def random_iterator(values: list[T]) -> Generator[T, None, None]:
    values_copy = values.copy()
    random.shuffle(values_copy)
    yield from values_copy


def random_extended_chunked_list(values: list[T], chunk_size: int, chunk_count: int) -> list[list[T]]:
    result_list = []
    it = iter([])
    while len(result_list) < chunk_count:
        chunk = []
        while len(chunk) < chunk_size:
            try:
                v = next(it)
            except StopIteration:
                it = random_iterator(values)
                v = next(it)
            chunk.append(v)
        result_list.append(chunk)

    return result_list
