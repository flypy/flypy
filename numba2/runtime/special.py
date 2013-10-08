# -*- coding: utf-8 -*-

"""
Special numba functions.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, overlay
from .lowlevel_impls import add_impl

from pykit import types as ptypes

__all__ = ['sizeof']

@jit('a -> int64', opaque=True)
def sizeof(obj):
    raise NotImplementedError("Not implemented at the python level")

@jit('a -> Type[a]')
def typeof(obj):
    raise NotImplementedError("Not implemented at the python level")

# ______________________________________________________________________
# Implementations

def implement_sizeof(builder, obj):
    return builder.ret(builder.sizeof(ptypes.Int64, [obj]))

add_impl(sizeof, "sizeof", implement_sizeof, ptypes.Int64)