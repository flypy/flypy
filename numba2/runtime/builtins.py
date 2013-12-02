# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import sys

try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from .. import jit, ijit, overlay, overload
from .interfaces import Sequence, Iterable, Iterator
from .obj.core import Range, List, Type, Complex
from .casting import cast
from numba2.types import int32, float64
from . import ffi

# ____________________________________________________________
# Type checking

@ijit('a -> Type[a] -> bool')
def isinstance(obj, type):
    return True

@jit('a -> Type[b] -> bool')
def isinstance(obj, type):
    raise NotImplementedError

# ____________________________________________________________
# Seq

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
# Strings

@jit
def str(x):
    return x.__str__()

@jit
def repr(x):
    return x.__repr__()

@jit
def unicode(x):
    return x.__unicode__()

@jit
def print(value, sep=' ', end='\n'):
    # TODO: *args and **kwargs
    s = str(value)
    ffi.libc.printf(s.buf.p) # TODO: Properties
    ffi.libc.printf(end.buf.p)
    #ffi.libc.puts(s.buf.p)

@jit
def clock():
    return ffi.libc.clock()

# ____________________________________________________________
# Conversion

if sys.version_info[0] < 3:
    @ijit
    def bool(x):
        return x.__nonzero__()
else:
    @ijit
    def bool(x):
        return x.__bool__()

# TODO: integral | floating typeset

@jit('a -> int64')
def int(x):
    return x.__int__()

@jit('a : integral -> float64')
def float(x):
    return cast(x, float64)

@jit('a : floating -> float64')
def float(x):
    return cast(x, float64)

# TODO: adjust type of constants depending on argtype

@jit('a : numeric -> a -> Complex[a]')
def complex(real, imag=0):
    return Complex(real, imag)

# ____________________________________________________________

@jit('a : numeric -> a')
def abs(x):
    if x < 0:
        x = -x
    return x

@jit('a -> b')
def abs(x):
    return x.__abs__()

# ____________________________________________________________

# TODO: Implement generator fusion

Py_ssize_t = 'int32' # TODO:

@ijit('int64 -> int64 -> int64 -> Range[]')
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
overlay(builtins.bool, bool)
overlay(builtins.int, int)
overlay(builtins.float, float)
overlay(builtins.complex, complex)
overlay(builtins.abs, abs)
overlay(builtins.range, range)
overlay(builtins.xrange, range)
overlay(builtins.list, list)
overlay(builtins.print, print)