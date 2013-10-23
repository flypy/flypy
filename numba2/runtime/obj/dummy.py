# -*- coding: utf-8 -*-

"""
Dummy type implementations.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from ... import jit
from blaze.datashape import Function as FunctionType
from ..conversion import ctype
from .pointerobject import Pointer

@jit(FunctionType())
class Function(object):
    layout = []

    @classmethod
    def ctype(cls, ty):
        restype = ctype(ty.restype)
        argtypes = [ctype(argtype) for argtype in ty.argtypes]
        return ctypes.POINTER(ctypes.CFUNCTYPE(restype, *argtypes))


@jit('ForeignFunction[restype, ...]')
class ForeignFunction(object):
    layout = [('p', 'Pointer[a]')]

    @staticmethod
    def fromobject(value, type):
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
        from numba2.cffi_support import is_cffi, ffi

        value = value.p

        #if is_cffi(value):
        #    value = ffi.addressof(value)
        #else:
        #    value = ctypes.pointer(value)

        return Pointer.toctypes(Pointer(value), Pointer[type])

@jit
class Void(object):
    layout = []

    @classmethod
    def ctype(cls, ty):
        return None # Sigh, ctypes
