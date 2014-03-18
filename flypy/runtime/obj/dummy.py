# -*- coding: utf-8 -*-

"""
Dummy type implementations.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from flypy import jit
from flypy.representation import byref
from flypy.conversion import ctype
from .pointerobject import Pointer, make_ctypes_ptr

from datashape import Function as FunctionType

#===------------------------------------------------------------------===
# Functions
#===------------------------------------------------------------------===

@jit(FunctionType())
class Function(object):
    layout = []

    # ----------------------

    @jit('a -> bool')
    def __nonzero__(self):
        return True

    __bool__ = __nonzero__

    # ----------------------

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

    # ----------------------

    @jit('a -> bool')
    def __nonzero__(self):
        return True

    __bool__ = __nonzero__

    # ----------------------

    @staticmethod
    def fromobject(value, type, keepalive):
        from flypy.support.cffi_support import is_cffi, ffi, is_cffi_func
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
        #return ctypes.CFUNCTYPE(restype, *argtypes)
        return ctypes.POINTER(ctypes.CFUNCTYPE(restype, *argtypes))

    @staticmethod
    def toctypes(value, type):
        value = value.p
        return make_ctypes_ptr(value, type)
        #ctype_func_p = Pointer.toctypes(Pointer(value), Pointer[type])
        #return ctype_func_p

# Set the 'varargs' property on the type of function types. This is
# somewhat of a gross hack, and clearly displays limitations in our
# type system
type_constructor = type

type_constructor(Function[None]).varargs = False
type_constructor(ForeignFunction[None]).varargs = False

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

    @staticmethod
    def toobject(value, type):
        return None