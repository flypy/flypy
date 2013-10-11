# -*- coding: utf-8 -*-

"""
Primitive operations, like 'a is b'.
"""

from __future__ import print_function, division, absolute_import

import operator

from .obj import NoneType
from .. import jit, typeof, overlay

#===------------------------------------------------------------------===
# Implementations
#===------------------------------------------------------------------===

@jit('a -> b -> bool')
def is_(a, b):
    return False

# TODO: Overload for variants !

@jit('NoneType -> NoneType -> bool')
def is_(a, b):
    return True

#===------------------------------------------------------------------===
# Overlays
#===------------------------------------------------------------------===

# We overlay operator.is_ with our own implementation. This works not only
# when operator.is_ is used in user-code, but frontend/translation.py itself
# turns 'is' operations into operator.is_ calls

overlay(operator.is_, is_)