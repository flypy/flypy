# -*- coding: utf-8 -*-

"""
Implement objects.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, ijit, typeof
from . import Void, Pointer
from . import librt as lib
from numba2.runtime.ffi import ffi

import cffi


#PyObject_p = typeof(lib.add).parameters[0]
PyObject_p = Pointer[Void[()]]

#===------------------------------------------------------------------===
# Implementation
#===------------------------------------------------------------------===

@jit
class Object(object):
    """
    Implement objects using using Cython functions which call the CPython C-API.
    """

    layout = [('obj', PyObject_p)]

    @ijit('a -> Pointer[void] -> void')
    def __init__(self, ptr):
        self.obj = ptr
        self.incref()

    @jit
    def incref(self):
        lib.Py_IncRef(self.obj)

    @jit
    def decref(self):
        lib.Py_DecRef(self.obj)

    @jit
    def __del__(self):
        self.decref()

    # ---------------------------------------

    @jit("a -> a")
    def __getiter__(self):
        return wrap(lib.getiter(self))

    @jit("a -> a")
    def __next__(self):
        return wrap(lib.next(self))

    #@jit("a -> a -> a")
    #def __getattr__(self, attr):
    #    return wrap(lib.getfield(self, attr))
    #
    #@jit("a -> a -> a -> a")
    #def __setattr__(self, attr, value):
    #    check(lib.setfield(self, attr, value))

    @jit("a -> a -> a")
    def __getitem__(self, idx):
        return wrap(lib.getitem(self, idx))

    @jit("a -> a -> a")
    def __setitem__(self, idx, value):
        check(lib.setitem(self, idx, value))

    # ---------------------------------------

    @ijit("a -> a -> a")
    def __add__(self, other):
        return wrap(lib.add(self.obj, other.obj))

    @jit("a -> a -> a")
    def __sub__(self, other):
        return wrap(lib.sub(self, other))

    @jit("a -> a -> a")
    def __mul__(self, other):
        return wrap(lib.mul(self, other))

    @jit("a -> a -> a")
    def __div__(self, other):
        return wrap(lib.divide(self, other))

    @jit("a -> a -> a")
    def __floordiv__(self, other):
        return wrap(lib.floordiv(self, other))

    @jit("a -> a -> a")
    def __lshift__(self, other):
        return wrap(lib.lshift(self, other))

    @jit("a -> a -> a")
    def __rshift__(self, other):
        return wrap(lib.rshift(self, other))

    @jit("a -> a -> a")
    def __bitor__(self, other):
        return wrap(lib.bitor(self, other))

    @jit("a -> a -> a")
    def __bitand__(self, other):
        return wrap(lib.bitand(self, other))

    # ---------------------------------------

    @jit("a -> a -> bool")
    def __lt__(self, other):
        return bool(wrap(lib.lt(self, other)))

    @jit("a -> a -> bool")
    def __le__(self, other):
        return bool(wrap(lib.le(self, other)))

    @jit("a -> a -> bool")
    def __gt__(self, other):
        return bool(wrap(lib.gt(self, other)))

    @jit("a -> a -> bool")
    def __ge__(self, other):
        return bool(wrap(lib.ge(self, other)))

    @jit("a -> a -> bool")
    def __eq__(self, other):
        return bool(wrap(lib.eq(self, other)))

    @jit("a -> a -> bool")
    def __ne__(self, other):
        return bool(wrap(lib.ne(self, other)))

    # ---------------------------------------

    @jit("a -> a")
    def __uadd__(self):
        return wrap(lib.uadd(self))

    @jit("a -> a")
    def __invert__(self):
        return wrap(lib.invert(self))

    @jit("a -> a")
    def __not___(self):
        return wrap(lib.not_(self))

    @jit("a -> a")
    def __usub__(self):
        return wrap(lib.usub(self))

    # ---------------------------------------

    @jit('a -> bool')
    def __nonzero__(self):
        result = wrap(lib.bool_(self))
        return cast(result, bool_)

    @jit('a -> bool')
    def __str__(self):
        result = wrap(lib.tostring(self))
        return cast(result, String)

    @jit('a -> bool')
    def __repr__(self):
        result = wrap(lib.torepr(self))
        return cast(result, String)

    # ---------------------------------------

    @classmethod
    def fromobject(cls, obj, ty):
        addr = ffi.cast('uintptr_t', id(obj))
        p = ffi.cast('void *', addr)
        return Object(p)

    @classmethod
    def toobject(cls, obj, ty):
        return ffi.cast('object', obj.obj)


@ijit
def wrap(p):
    check(p)
    return Object(p)

@ijit
def check(p):
    return p # TODO:
