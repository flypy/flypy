# -*- coding: utf-8 -*-

"""
Implement casting (cast function).
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit
from numba2.runtime import Int, Float, Type
from numba2.runtime.lowlevel_impls import add_impl, lltype

castmap = {
    Int: int,
    Float: float,
}

@jit('a : numeric -> Type[b : numeric] -> b', opaque=True)
def cast(x, type):
    """Cast a value of type a to type b"""
    # Pure python implementations
    cls = type.impl
    caster = castmap[cls]
    return caster(x)


def convert(builder, argtypes, value, type):
    valtype, typetype = argtypes # e.g. `int, Type[double]`
    type = typetype.parameters[0]
    builder.ret(builder.convert(lltype(type), value))


argtype = lambda argtypes: lltype(argtypes[1].parameters[0])
add_impl(cast, "cast", convert, restype_func=argtype)
