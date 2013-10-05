# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

__all__ = ['Function', 'Type', 'Bool', 'Int', 'Float']

#===------------------------------------------------------------------===
# Types
#===------------------------------------------------------------------===

from blaze import dshape
from blaze.datashape import free, TypeVar, TypeConstructor
from blaze.datashape import Mono as Type
from .runtime.obj import Function, Pointer, Bool, Int, Float, Void

#===------------------------------------------------------------------===
# Units
#===------------------------------------------------------------------===

bool_   = Bool[()]
void    = Void[()]
int32   = Int[32, False]
int64   = Int[64, False]
float32 = Float[32]
float64 = Float[64]
