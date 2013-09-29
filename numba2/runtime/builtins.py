# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from .. import jit, ijit, typedef, overload

# ____________________________________________________________

@ijit('Iterable[a] -> Iterator[a]')
def iter(x):
    return x.__iter__()

@ijit('Iterator[a] -> a')
def next(x):
    return x.__next__()

# ____________________________________________________________

@ijit #('Int -> Int -> Int -> Int')
def len_range(start, stop, step):
    if step < 0:
        start, stop, step = stop, start, -step
    if start <= stop:
        return 0
    return (stop - start - 1) // step + 1

@jit('Int -> Int -> Int -> Iterable[Int]')
def range(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0

    length = len_range(start, stop, step)
    i = 0
    while i < length:
        yield start
        start += step

# ____________________________________________________________

typedef(builtins.iter, iter)
typedef(builtins.next, next)
typedef(builtins.range, range)