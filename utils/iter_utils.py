class IncompleteLastChunk(Exception):
    pass


class IterableExceed(Exception):
    pass


def limited_iterator(iterable, limit: int):
    while limit:
        yield next(iterable)
        limit -= 1


def get_array_chunks(ndarray, size, count=None):
    chunks = (ndarray[pos:min(pos + size, len(ndarray))] for pos in range(0, len(ndarray), size))
    if count:
        yield from limited_iterator(chunks, count)
    else:
        yield from chunks


def get_chunks(iterable, size, count=None):
    def generator():
        buff = []
        while True:
            try:
                v = next(iterable)
            except StopIteration:
                if buff:
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