# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from .. import jit, ijit, overlay, overload

# ____________________________________________________________

@jit('Iterable[a] -> Iterator[a]')
def iter(x):
    return x.__iter__()

@jit('Iterator[a] -> a')
def next(x):
    return x.__next__()

# ____________________________________________________________

@jit
def len_range(start, stop, step):
    if step < 0:
        start, stop, step = stop, start, -step
    if stop <= start:
        return 0
    return (stop - start - 1) // step + 1

@jit #('Int -> Int -> Int -> Iterable[Int]')
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

overlay(builtins.iter, iter)
overlay(builtins.next, next)
overlay(builtins.range, range)