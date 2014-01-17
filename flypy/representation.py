# -*- coding: utf-8 -*-

"""
Value representation of instances of user-defined types.
"""

from __future__ import print_function, division, absolute_import

#===------------------------------------------------------------------===
# Object Representation
#===------------------------------------------------------------------===

def c_primitive(type):
    from flypy import types

    return type.impl in (types.Bool, types.Int, types.Float, types.Pointer,
                         types.Void, types.Function)

def stack_allocate(type):
    """
    Determine whether values of this type should be stack-allocated and partake
    directly as values under composition.
    """
    return type.impl.stackallocate

def byref(type):
    return stack_allocate(type) and not c_primitive(type)
