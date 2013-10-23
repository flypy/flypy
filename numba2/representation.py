# -*- coding: utf-8 -*-

"""
Value representation of instances of user-defined types.
"""

from __future__ import print_function, division, absolute_import
from numba2.runtime.conversion import ctype
from pykit.utils.ctypes import from_ctypes_type

#===------------------------------------------------------------------===
# Type Representation
#===------------------------------------------------------------------===

def representation_type(ty):
    """
    Get the low-level representation type for a high-level (user-defined) type.
    """
    cty = ctype(ty)
    return from_ctypes_type(cty)