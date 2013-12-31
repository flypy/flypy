# -*- coding: utf-8 -*-

"""
Type coercion.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, ijit, cjit, Type, Pointer, void, cast

jit = cjit

#===------------------------------------------------------------------===
# coerce
#===------------------------------------------------------------------===

#@jit('a -> Type[a] -> a')
#def coerce(x, ty):
#    return x

# -- bool

@jit('a -> Type[bool] -> bool')
def coerce(x, ty):
    return bool(x)

# -- numeric

@jit('a : integral -> Type[b : integral] -> b')
def coerce(x, ty):
    return cast(x, ty)

@jit('a -> Type[b : integral] -> b')
def coerce(x, ty):
    return int(x)

@jit('a : floating -> Type[b : integral] -> b')
def coerce(x, ty):
    raise TypeError("Can't coerce float to int")

@jit('a -> Type[b : floating] -> b')
def coerce(x, ty):
    return cast(x, ty)

@jit('a -> Type[b : complexes] -> b')
def coerce(x, ty):
    return complex(x)

# -- pointer

@jit('Pointer[a] -> Type[Pointer[void]] -> Pointer[void]')
def coerce(x, ty):
    return cast(x, ty)
