# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba2 import jit, ijit
from ..interfaces import Sequence, Iterable, Iterator

# ________________________________________________________________

# TODO: Implement generator fusion

Py_ssize_t = 'int32' # TODO:

@jit
def len_range(start, stop, step):
    if step < 0:
        start, stop, step = stop, start, -step
    if stop <= start:
        return 0
    return (stop - start - 1) // step + 1

@jit
class Range(Sequence):

    layout = [('start', Py_ssize_t), ('stop', Py_ssize_t), ('step', Py_ssize_t)]

    @ijit
    def __iter__(self):
        return RangeIterator(self.start, self.step, len(self))

    @ijit
    def __len__(self):
        return len_range(self.start, self.stop, self.step)

    @jit('a -> bool')
    def __nonzero__(self):
        return bool(len(self))


@jit
class RangeIterator(Iterator):

    layout = [('start', Py_ssize_t), ('step', Py_ssize_t), ('length', Py_ssize_t)]

    @ijit
    def __next__(self):
        if self.length > 0:
            result = self.start
            self.start += self.step
            self.length -= 1
            return result
        raise StopIteration

# ________________________________________________________________