# -*- coding: utf-8 -*-

"""
Special numba functions.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, Type, Pointer
from numba2.pipeline import fresh_env, phase
from numba2.runtime import lowlevel_impls
from numba2.compiler import opaque

__all__ = ['typeof']

@jit('a -> Type[a]', opaque=True)
def typeof(obj):
    raise NotImplementedError("Not implemented at the python level")

@jit('a -> Pointer[void]', opaque=True)
def addressof(func):
    raise NotImplementedError("Not implemented at the python level")

#===------------------------------------------------------------------===
# Low-level implementations
#===------------------------------------------------------------------===

## typeof()

def make_typeof(py_func, argtypes):
    [type] = argtypes

    @jit
    def typeof_impl(obj):
        return type

    env = fresh_env(typeof_impl, tuple(argtypes), "cpu")
    func, env = phase.opt(typeof_impl, env)
    return func

opaque.implement_opaque(typeof, make_typeof)

## addressof()

def impl_addressof(builder, argtypes, obj):
    builder.ret(builder.addressof(obj))

lowlevel_impls.add_impl(addressof, "addressof", impl_addressof)