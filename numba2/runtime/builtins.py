# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from .. import jit, ijit, overlay, overload
from .interfaces import Sequence, Iterable, Iterator
from .type import Type

# ____________________________________________________________

@ijit('a -> Type[a] -> bool')
def isinstance(obj, type):
    return True

@jit('a -> Type[b] -> bool')
def isinstance(obj, type):
    raise NotImplementedError

# ____________________________________________________________

# TODO: Type join!

@ijit #('Iterable[a] -> Iterator[a]')
def iter(x):
    return x.__iter__()

@ijit #('Iterator[a] -> a')
def next(x):
    return x.__next__()

@ijit #('Sequence[a] -> Py_ssize_t')
def len(x):
    return x.__len__()

# ____________________________________________________________

# TODO: Implement generator fusion

Py_ssize_t = 'int32' # TODO:

@jit
def len_range(start, stop, step):
    if step < 0:
        start, stop, step = stop, start, -step
    if stop <= start:
        return 0
    return (stop - start - 1) // step + 1

@ijit
def range(start, stop=0xdeadbeef, step=1):
    # TODO: We need to either optimize variants, or recognize that
    # 'x is None' is equivalent to isinstance(x, NoneType) and prune the
    # alternative branch during type inference
    if stop == 0xdeadbeef:
        stop = start
        start = 0

    return Range(start, stop, step)

@jit
class Range(Sequence):

    layout = [('start', Py_ssize_t), ('stop', Py_ssize_t), ('step', Py_ssize_t)]

    @ijit
    def __iter__(self):
        return RangeIterator(self.start, self.step, len(self))

    @ijit
    def __len__(self):
        return len_range(self.start, self.stop, self.step)


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

# ____________________________________________________________

overlay(builtins.isinstance, isinstance)
overlay(builtins.iter, iter)
overlay(builtins.next, next)
overlay(builtins.len, len)
overlay(builtins.range, range)