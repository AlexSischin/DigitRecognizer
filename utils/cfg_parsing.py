from typing import Callable, TypeVar

T = TypeVar('T')


def to_tuple(map_func: Callable[[str], T]) -> Callable[[str], tuple[T, ...]]:
    def func(s: str) -> tuple[T, ...]:
        int_strings = s.split(' ')
        return tuple([map_func(e) for e in int_strings])

    return func


# Factory dict
def to_tuple_f_dict(f_dict: dict[str, Callable[[], T]]) -> Callable[[str], tuple[T, ...]]:
    def func(s: str) -> tuple[T, ...]:
        int_strings = s.split(' ')
        return tuple([f_dict[e]() for e in int_strings])

    return func


# Object dict
def to_tuple_o_dict(o_dict: dict[str, T]) -> Callable[[str], tuple[T, ...]]:
    def func(s: str) -> tuple[T, ...]:
        int_strings = s.split(' ')
        return tuple([o_dict[e] for e in int_strings])

    return func
