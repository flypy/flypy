# -*- coding: utf-8 -*-

"""
Dummy type implementations.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from ... import jit
from blaze.datashape import Function as FunctionType
from ..conversion import ctype, byref
from .pointerobject import Pointer

#===------------------------------------------------------------------===
# Functions
#===------------------------------------------------------------------===

@jit(FunctionType())
class Function(object):
    layout = []

    @classmethod
    def ctype(cls, ty):
        restype = ctype(ty.restype)
        argtypes = [ctype(argtype) for argtype in ty.argtypes]

        if byref(ty.restype):
            argtypes.append(ctypes.POINTER(restype))
            restype = None # void

        return ctypes.POINTER(ctypes.PYFUNCTYPE(restype, *argtypes))


@jit('ForeignFunction[restype, ...]')
class ForeignFunction(object):
    layout = [('p', 'Pointer[a]')]

    @staticmethod
    def fromobject(value, type):
        from numba2.cffi_support import is_cffi, ffi, is_cffi_func
        #if is_cffi(value) and is_cffi_func(value):
        #    value = ffi.addressof(value)
        return ForeignFunction(value)

    @staticmethod
    def toobject(value, type):
        return value.p

    @classmethod
    def ctype(cls, ty):
        restype = ctype(ty.parameters[-1])
        argtypes = [ctype(argtype) for argtype in ty.parameters[:-1]]
        return ctypes.CFUNCTYPE(restype, *argtypes)

    @staticmethod
    def toctypes(value, type):
        value = value.p
        return Pointer.toctypes(Pointer(value), Pointer[type])

#===------------------------------------------------------------------===
# Void
#===------------------------------------------------------------------===

@jit
class Void(object):
    layout = []

    @classmethod
    def ctype(cls, ty):
        return None # Sigh, ctypes

    @classmethod
    def toctypes(cls, value, ty):
        return None

#===------------------------------------------------------------------===
# NULL
#===------------------------------------------------------------------===

void = Void[()]
_NULL = ctypes.c_void_p(0)

@jit
class NULL(object):
    layout = []

    @jit('a -> Pointer[b] -> bool')
    def __eq__(self, other):
        p = cast(other, Pointer[void])
        return p == _NULL

    #@jit('a -> b -> bool')
    #def __eq__(self, other):
    #    return False
