# -*- coding: utf-8 -*-

"""
Implement objects.
"""

from __future__ import print_function, division, absolute_import
import ctypes

import flypy.types
from flypy import jit, ijit, typeof
from flypy.runtime.obj.stringobject import as_cstring
from flypy.coretypes import Void, Pointer, String
from flypy.runtime.lib import librt as lib

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

    layout = [('ptr', PyObject_p)]

    @ijit('a -> Pointer[void] -> void')
    def __init__(self, ptr):
        self.ptr = ptr
        self.incref()

    @jit
    def incref(self):
        lib.Py_IncRef(self.ptr)

    @jit
    def decref(self):
        lib.Py_DecRef(self.ptr)

    @jit
    def __del__(self):
        self.decref()

    # ---------------------------------------

    @jit("a -> a")
    def __getiter__(self):
        return wrap(lib.getiter(self.ptr))

    @jit("a -> a")
    def __next__(self):
        return wrap(lib.next(self.ptr))

    @jit("a -> String[] -> a")
    def __getattr__(self, attr):
        attr = as_cstring(attr)
        return wrap(lib.getfield(self.ptr, attr))

    @jit("a -> String[] -> a -> void")
    def __setattribute__(self, attr, value):
        attr = as_cstring(attr)
        check(lib.setfield(self.ptr, attr, value.ptr))

    @jit("a -> a -> a")
    def __getitem__(self, idx):
        return wrap(lib.getitem(self.ptr, idx.ptr))

    @jit("a -> a -> a")
    def __setitem__(self, idx, value):
        check(lib.setitem(self.ptr, idx.ptr, value.ptr))

    # ---------------------------------------

    @ijit #("a -> a -> a")
    def __add__(self, other):
        return wrap(lib.add(self.ptr, other.ptr))

    @jit("a -> a -> a")
    def __sub__(self, other):
        return wrap(lib.sub(self.ptr, other.ptr))

    @jit("a -> a -> a")
    def __mul__(self, other):
        return wrap(lib.mul(self.ptr, other.ptr))

    @jit("a -> a -> a")
    def __div__(self, other):
        return wrap(lib.divide(self.ptr, other.ptr))

    @jit("a -> a -> a")
    def __floordiv__(self, other):
        return wrap(lib.floordiv(self.ptr, other.ptr))

    @jit("a -> a -> a")
    def __lshift__(self, other):
        return wrap(lib.lshift(self.ptr, other.ptr))

    @jit("a -> a -> a")
    def __rshift__(self, other):
        return wrap(lib.rshift(self.ptr, other.ptr))

    @jit("a -> a -> a")
    def __bitor__(self, other):
        return wrap(lib.bitor(self.ptr, other.ptr))

    @jit("a -> a -> a")
    def __bitand__(self, other):
        return wrap(lib.bitand(self.ptr, other.ptr))

    # ---------------------------------------

    @jit("a -> a -> bool")
    def __lt__(self, other):
        return bool(wrap(lib.lt(self.ptr, other.ptr)))

    @jit("a -> a -> bool")
    def __le__(self, other):
        return bool(wrap(lib.le(self.ptr, other.ptr)))

    @jit("a -> a -> bool")
    def __gt__(self, other):
        return bool(wrap(lib.gt(self.ptr, other.ptr)))

    @jit("a -> a -> bool")
    def __ge__(self, other):
        return bool(wrap(lib.ge(self.ptr, other.ptr)))

    @jit("a -> a -> bool")
    def __eq__(self, other):
        return bool(wrap(lib.eq(self.ptr, other.ptr)))

    @jit("a -> a -> bool")
    def __ne__(self, other):
        return bool(wrap(lib.ne(self.ptr, other.ptr)))

    # ---------------------------------------

    @jit("a -> a")
    def __uadd__(self):
        return wrap(lib.uadd(self.ptr))

    @jit("a -> a")
    def __invert__(self):
        return wrap(lib.invert(self.ptr))

    @jit("a -> a")
    def __not___(self):
        return wrap(lib.not_(self.ptr))

    @jit("a -> a")
    def __usub__(self):
        return wrap(lib.usub(self.ptr))

    # ---------------------------------------

    @jit('a -> bool')
    def __nonzero__(self):
        return lib.istrue(self.ptr)

    @jit('a -> int64')
    def __len__(self):
        return lib.length(self.ptr)

    # ---------------------------------------

    @jit
    def __str__(self):
        # Object -> str Object
        str_obj = wrap(lib.tostring(self.ptr))
        # Get pointer to str buf
        p = lib.asstring(str_obj.ptr)

        # Build flypy String
        n = len(str_obj)
        buf = flypy.types.Buffer(p, n + 1, False)
        return flypy.types.String(buf)

    @jit
    def __repr__(self):
        result = wrap(lib.torepr(self))
        return str(result)

    # ---------------------------------------

    @classmethod
    def fromobject(cls, obj, ty):
        addr = lib.address(obj)
        p = ctypes.c_void_p(addr)
        return Object(p)

    @classmethod
    def toobject(cls, obj, ty):
        return lib.fromvoidp(obj.ptr)


@ijit
def wrap(p):
    check(p)
    return Object(p)

@ijit
def check(p):
    return p # TODO:
