# -*- coding: utf-8 -*-

"""
Some test support utilities.
"""

from __future__ import print_function, division, absolute_import

from pykit.utils.ctypes import is_ctypes_struct_type, is_ctypes_pointer_type

class CTypesStruct(object):
    """
    Wrap ctypes structs/pointers to structs for uniform access.
    """

    def __init__(self, ctypes_val):
        if is_ctypes_pointer_type(type(ctypes_val)):
            ctypes_val = ctypes_val[0]
        self.val = ctypes_val

    def __getattr__(self, attr):
        result = getattr(self.val, attr)
        cty = type(result)
        if is_ctypes_pointer_type(cty) or is_ctypes_struct_type(cty):
            result = CTypesStruct(result)
        return result

    def __repr__(self):
        return "CTypesStruct(%s)" % (self.val,)