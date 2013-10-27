# -*- coding: utf-8 -*-

"""
Implement casting (cast function).
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit
from numba2.runtime import Int, Float, Type, Pointer
from numba2.runtime.lowlevel_impls import add_impl, lltype

#===------------------------------------------------------------------===
# Python
#===------------------------------------------------------------------===

castmap = {
    Int: int,
    Float: float,
}

def py_cast(x, type):
    # Pure python implementations
    cls = type.impl
    caster = castmap[cls]
    return caster(x)

#===------------------------------------------------------------------===
# Numba Casting
#===------------------------------------------------------------------===

@jit('a : numeric -> Type[b : numeric] -> b', opaque=True)
def cast(x, type):
    """Cast a value of type a to type b"""
    return py_cast(x, type) # pure python

@jit('Pointer[a] -> Type[Pointer[b]] -> Pointer[b]', opaque=True)
def cast(x, type):
    return py_cast(x, type) # pure python

@jit('Pointer[a] -> Type[int64] -> int64', opaque=True)
def cast(x, type):
    return py_cast(x, type) # pure python

#===------------------------------------------------------------------===
# Low-level implementation
#===------------------------------------------------------------------===

def convert(builder, argtypes, value, type):
    valtype, typetype = argtypes # e.g. `int, Type[double]`
    type = typetype.parameters[0]

    if type.impl == Pointer or (type.impl == Int and valtype.impl == Pointer):
        result = builder.ptrcast(lltype(type), value)
    else:
        result = builder.convert(lltype(type), value)

    builder.ret(result)


def result_type(argtypes):
    return lltype(argtypes[1].parameters[0])

add_impl(cast, "cast", convert, restype_func=result_type)
