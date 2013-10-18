# -*- coding: utf-8 -*-

"""
Dummy type implementations.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from ... import jit
from blaze.datashape import Function as FunctionType
from ..conversion import ctype

@jit(FunctionType())
class Function(object):
    layout = []

    @classmethod
    def ctype(cls, ty):
        restype = ctype(ty.restype)
        argtypes = [ctype(argtype) for argtype in ty.argtypes]
        return ctypes.CFUNCTYPE(restype, *argtypes)


@jit
class Void(object):
    layout = []

    @classmethod
    def ctype(cls, ty):
        return None # Sigh, ctypes