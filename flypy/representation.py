# -*- coding: utf-8 -*-

"""
Value representation of instances of user-defined types.
"""

from __future__ import print_function, division, absolute_import

import flypy as nb

#===------------------------------------------------------------------===
# Object Representation
#===------------------------------------------------------------------===

def c_primitive(type):
    return type.impl in (nb.Bool, nb.Int, nb.Float, nb.Pointer, nb.Void,
                         nb.Function, nb.ForeignFunction)

def stack_allocate(type):
    """
    Determine whether values of this type should be stack-allocated and partake
    directly as values under composition.
    """
    return type.impl.stackallocate

def byref(type):
    return stack_allocate(type) and not c_primitive(type)
