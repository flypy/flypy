# -*- coding: utf-8 -*-

"""
vector implementation.
"""

from __future__ import print_function, division, absolute_import

import numba2
from numba2 import sjit, jit
from numba2.compiler import representation_type, lltype
from numba2.conversion import ctype
from numba2.runtime import formatting
from ..interfaces import Number
from ..lowlevel_impls import add_impl_cls
from pykit import types as ptypes

#
# from numba2 import jit
# from llvm.core import Constant
# from ..lowlevel_impls import add_impl_cls
# from pykit import types as ptypes
# from ..interfaces import Number

@sjit('Vector[base, count]')
class Vector(Number):
    layout = []

    @jit('Vector[base, count] -> int64 -> base', opaque=True)
    def __getitem__(self, index):
        pass

    @jit('Vector[base, count] -> int64 -> base', opaque=True)
    def __setitem__(self, key, value):
        pass

    @jit('Vector[base, count] -> int64', opaque=True)
    def __len__(self):
        pass

# constructors?

#@jit('Array[base, count] -> Vector[base, count]', opaque=True)
#def pack(builder, argtypes, *args):
#    builder.ret(Constant.vector(*args))

#@jit('base -> Vector[base, count]', opaque=True)
#def fill(builder, argtypes, value):
#    pack(builder, argtypes, *([value] * argtypes[0].parameters[1]))



# low level operations

def vector_getitem(builder, argtypes, vector, idx):
    builder.ret(builder.extractelement(vector, idx))

def vector_setitem(builder, argtypes, vector, value, idx):
    builder.ret(builder.insertelement(vector, value, idx))

def vector_len(builder, argtypes, vector):
    builder.ret(argtypes[0].parameters[1])

# implementations

add_impl_cls(Vector, "__getitem__", vector_getitem, restype_func=lambda argtypes: lltype(argtypes[0].parameters[0]))
add_impl_cls(Vector, "__setitem__", vector_setitem, restype_func=lambda argtypes: argtypes[0])
add_impl_cls(Vector, "__len__", vector_len, restype=ptypes.UInt64)

#===------------------------------------------------------------------===
# typeof
#===------------------------------------------------------------------===

#@typeof.case(Vector)
#def typeof(vec):
#    return vec
