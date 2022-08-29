from typing import TypeVar, Iterable, Generator, Tuple

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")
_T4 = TypeVar("_T4")
_T5 = TypeVar("_T5")


def zip2(vals1: Iterable[_T1] = None,
         vals2: Iterable[_T2] = None
         ) -> Generator[Tuple[_T1, _T2], None, None]:
    for vals in zip(vals1, vals2):
        yield vals


def zip3(vals1: Iterable[_T1] = None,
         vals2: Iterable[_T2] = None,
         vals3: Iterable[_T3] = None
         ) -> Generator[Tuple[_T1, _T2, _T3], None, None]:
    for vals in zip(vals1, vals2, vals3):
        yield vals


def zip4(vals1: Iterable[_T1] = None,
         vals2: Iterable[_T2] = None,
         vals3: Iterable[_T3] = None,
         vals4: Iterable[_T4] = None
         ) -> Generator[Tuple[_T1, _T2, _T3, _T4], None, None]:
    for vals in zip(vals1, vals2, vals3, vals4):
        yield vals


def zip5(vals1: Iterable[_T1] = None,
         vals2: Iterable[_T2] = None,
         vals3: Iterable[_T3] = None,
         vals4: Iterable[_T4] = None,
         vals5: Iterable[_T5] = None
         ) -> Generator[Tuple[_T1, _T2, _T3, _T4, _T5], None, None]:
    for vals in zip(vals1, vals2, vals3, vals4, vals5):
        yield vals
