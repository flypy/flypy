# -*- coding: utf-8 -*-

"""
Object layout.
"""

from __future__ import print_function, division, absolute_import

from flypy import conversion

from pykit import types as ptypes
from pykit.utils import ctypes_support

#===------------------------------------------------------------------===
# Types
#===------------------------------------------------------------------===

def representation_type(ty):
    """
    Get the low-level representation type for a high-level (user-defined) type.

    Returns
    =======
    The pykit type for the object layout.
    """

    from flypy.lib import vectorobject
    from flypy.lib import arrayobject
    from flypy.runtime.obj import pointerobject

    # NOTE: special cases should be kept to an absolute minimum here. They
    #       should probably be introduced only if ctypes cannot represent the
    #       type
    if ty.impl == vectorobject.Vector:
        base, count = ty.parameters
        return ptypes.Vector(representation_type(base), count)
    elif ty.impl == pointerobject.Pointer:
        # type pointed to may not be supported by ctypes
        (base,) = ty.parameters
        if base.impl == vectorobject.Vector:
            return ptypes.Pointer(representation_type(base))

    cty = conversion.ctype(ty)
    result_type = ctypes_support.from_ctypes_type(cty)

    # struct uses pointer
    if result_type.is_struct:
        result_type = ptypes.Pointer(result_type)

    return result_type
