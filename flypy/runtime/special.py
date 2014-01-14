# -*- coding: utf-8 -*-

"""
Special flypy functions.
"""

from __future__ import print_function, division, absolute_import

import flypy
from flypy import jit, Type, Pointer, String, overlay
from flypy.pipeline import fresh_env, phase
from flypy.runtime import lowlevel_impls
from flypy.compiler import opaque

from pykit import types as ptypes
from pykit.ir import Op

__all__ = ['typeof']

@jit('a -> Type[a]', opaque=True)
def typeof(obj):
    """
    Take the type of a given object. This first executes the sub-expression!
    """
    raise NotImplementedError("Not implemented at the python level")

#@jit('a -> Pointer[void]', opaque=True)
#def addressof(func):
#    """
#    Take the address of a given function.
#    """
#    raise NotImplementedError("Not implemented at the python level")

#===------------------------------------------------------------------===
# Debug printing...
#===------------------------------------------------------------------===

@jit('a -> void', opaque=True)
def debugprint(expr):
    """
    Print information about `expr` during type inferencing.
    """
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
    func, env = phase.ll_lower(typeof_impl, env)
    return func

opaque.implement_opaque(typeof, make_typeof)

## addressof()

#def impl_addressof(builder, argtypes, obj):
#    builder.ret(builder.addressof(obj))
#
#lowlevel_impls.add_impl(addressof, "addressof", impl_addressof)

## Overlays

# make flypy.typeof() available in flypy code
overlay(flypy.typeof, typeof)