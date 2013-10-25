# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from .. import jit, ijit, overlay, overload
from .interfaces import Sequence, Iterable, Iterator
from .obj import Range, List, Type
from .casting import cast
from numba2.types import int32, float64

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

@ijit
def str(x):
    return x.__str__()

@ijit
def repr(x):
    return x.__repr__()

@ijit
def unicode(x):
    return x.__unicode__()

@ijit
def print(value, sep=' ', end='\n'):
    # TODO: *args and **kwargs
    raise NotImplementedError

# ____________________________________________________________

@jit('a : numeric -> int32')
def int(x):
    return cast(x, int32)

@jit('a : numeric -> float64')
def float(x):
    return cast(x, float64)

# ____________________________________________________________

# TODO: Implement generator fusion

Py_ssize_t = 'int32' # TODO:

@ijit
def range(start, stop=0xdeadbeef, step=1):
    # TODO: We need to either optimize variants, or recognize that
    # 'x is None' is equivalent to isinstance(x, NoneType) and prune the
    # alternative branch during type inference
    if stop == 0xdeadbeef:
        stop = start
        start = 0

    return Range(start, stop, step)

# ____________________________________________________________

@ijit('Iterable[x] -> List[x]')
def list(value):
    result = []
    result.extend(value)
    return result

# TODO: Overloading on arity
#@ijit
#def list():
#    return []

# ____________________________________________________________

overlay(builtins.isinstance, isinstance)
overlay(builtins.iter, iter)
overlay(builtins.next, next)
overlay(builtins.len, len)
overlay(builtins.str, str)
overlay(builtins.repr, repr)
overlay(builtins.unicode, unicode)
overlay(builtins.int, int)
overlay(builtins.float, float)
overlay(builtins.range, range)
overlay(builtins.list, list)