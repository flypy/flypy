# -*- coding: utf-8 -*-

"""
Implement casting (cast function).
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit
from numba2.compiler import lltype
from numba2.runtime.lowlevel_impls import add_impl
from numba2.runtime.obj.core import Int, Float, Type, Pointer

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
def numeric_cast(x, type):
    """Cast a value of type a to type b"""
    return py_cast(x, type) # pure python

@jit('Pointer[a] -> Type[Pointer[b]] -> Pointer[b]', opaque=True)
def bitcast(x, type):
    return py_cast(x, type) # pure python

@jit('int64 -> Type[Pointer[a]] -> Pointer[a]', opaque=True)
def inttoptr(x, type):
    return py_cast(x, type) # pure python

@jit('Pointer[a] -> Type[int64] -> int64', opaque=True)
def ptrtoint(x, type):
    return py_cast(x, type) # pure python

#===------------------------------------------------------------------===
# cast()
#===------------------------------------------------------------------===

@jit('a : numeric -> Type[b : numeric] -> b')
def cast(x, type):
    """Cast a value of type a to type b"""
    return numeric_cast(x, type)

@jit('Pointer[a] -> Type[Pointer[b]] -> Pointer[b]')
def cast(x, type):
    return bitcast(x, type)

@jit('int64 -> Type[Pointer[a]] -> Pointer[a]')
def cast(x, type):
    return inttoptr(x, type)

@jit('Pointer[a] -> Type[int64] -> int64')
def cast(x, type):
    return ptrtoint(x, type)

#===------------------------------------------------------------------===
# Low-level implementation
#===------------------------------------------------------------------===

def numeric_cast_impl(builder, argtypes, value, type):
    convert_impl(builder, argtypes, value, type)

def bitcast_impl(builder, argtypes, value, type):
    ptrcast(builder, argtypes, value, type)

def inttoptr_impl(builder, argtypes, value, type):
    convert_impl(builder, argtypes, value, type)

def ptrtoint_impl(builder, argtypes, value, type):
    ptrcast(builder, argtypes, value, type)

# ---------- helper -------------

def convert_impl(builder, argtypes, value, type):
    valtype, typetype = argtypes # e.g. `int, Type[double]`
    type = typetype.parameters[0]
    result = builder.convert(lltype(type), value)
    builder.ret(result)

def ptrcast(builder, argtypes, value, type):
    valtype, typetype = argtypes # e.g. `int, Type[double]`
    type = typetype.parameters[0]
    result = builder.ptrcast(lltype(type), value)
    builder.ret(result)

# ---------- add impls -------------

def result_type(argtypes):
    return lltype(argtypes[1].parameters[0])

add_impl(numeric_cast, "numeric_cast", numeric_cast_impl,
         restype_func=result_type)
add_impl(bitcast, "bitcast", bitcast_impl, restype_func=result_type)
add_impl(inttoptr, "inttoptr", inttoptr_impl, restype_func=result_type)
add_impl(ptrtoint, "ptrtoint", ptrtoint_impl, restype_func=result_type)