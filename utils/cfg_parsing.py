from typing import Callable, TypeVar

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


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


def to_dict(key_map_func: Callable[[str], K],
            value_map_func: Callable[[str], V],
            nullable=True
            ) -> Callable[[str], dict[K, V]]:
    if nullable:
        key_map_func = nullable_map(key_map_func)
        value_map_func = nullable_map(value_map_func)

    def func(s: str) -> dict[K, V]:
        kv_strings = s.split(' ')
        kv_tuples = [kv.split(':') for kv in kv_strings]
        return {key_map_func(k): value_map_func(v) for k, v in kv_tuples}

    return func


def nullable_map(map_func: Callable[[str], T]) -> Callable[[str], T | None]:
    def func(s: str):
        return map_func(s) if s != 'None' else None

    return func
