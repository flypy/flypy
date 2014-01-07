# -*- coding: utf-8 -*-

"""
Support for ctypes.
"""

from __future__ import print_function, division, absolute_import

import ctypes.util

from numba2 import coretypes
from numba2.lib import arrayobject
from pykit.utils import hashable
from pykit.utils.ctypes_support import is_ctypes_struct_type, is_ctypes_pointer_type, is_ctypes_array_type

#===------------------------------------------------------------------===
# CTypes Utils
#===------------------------------------------------------------------===

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

#===------------------------------------------------------------------===
# CTypes Types for Type Checking
#===------------------------------------------------------------------===

libc = ctypes.CDLL(ctypes.util.find_library('c'))

_ctypes_scalar_type = type(ctypes.c_int)
_ctypes_func_type = (type(ctypes.CFUNCTYPE(ctypes.c_int)), type(libc.printf))
_ctypes_pointer_type = type(ctypes.POINTER(ctypes.c_int))
_ctypes_array_type = type(ctypes.c_int * 2)

CData = type(ctypes.c_int(10)).__mro__[-2]

#===------------------------------------------------------------------===
# Check Whether values are ctypes values
#===------------------------------------------------------------------===

def is_ctypes_function_type(value):
    return isinstance(value, _ctypes_func_type)

def is_ctypes_function(value):
    return is_ctypes_function_type(type(value))

def is_ctypes_value(ctypes_value):
    return isinstance(ctypes_value, CData)

def is_ctypes_struct_type(ctypes_type):
    return (isinstance(ctypes_type, type) and
            issubclass(ctypes_type, ctypes.Structure))

def is_ctypes_pointer_type(ctypes_type):
    return isinstance(ctypes_type, _ctypes_pointer_type)

def is_ctypes_type(ctypes_type):
    return (
       (isinstance(ctypes_type, _ctypes_scalar_type)) or
       is_ctypes_struct_type(ctypes_type)
    )

def is_ctypes(value):
    "Check whether the given value is a ctypes value"
    return is_ctypes_value(value) or is_ctypes_type(value)

ptrval = lambda val: ctypes.cast(val, ctypes.c_void_p).value

#===------------------------------------------------------------------===
# Type mapping (ctypes -> numba)
#===------------------------------------------------------------------===

ctypes_map = {
    ctypes.c_bool :  coretypes.bool_,
    ctypes.c_char :  coretypes.int8,
    ctypes.c_int8 :  coretypes.int8,
    ctypes.c_int16:  coretypes.int16,
    ctypes.c_int32:  coretypes.int32,
    ctypes.c_int64:  coretypes.int64,
    ctypes.c_uint8 : coretypes.uint8,
    ctypes.c_uint16: coretypes.uint16,
    ctypes.c_uint32: coretypes.uint32,
    ctypes.c_uint64: coretypes.uint64,
    ctypes.c_float:  coretypes.float32,
    ctypes.c_double: coretypes.float64,
    None:            coretypes.void,
    ctypes.c_char_p: coretypes.Pointer[coretypes.char],
}

def from_ctypes_type(cty, ctypes_value=None):
    """
    Convert a ctypes type to a numba type.

    Supported are structs, unit types (int/float)
    """
    if hashable(cty) and cty in ctypes_map:
        return ctypes_map[cty]
    elif cty is ctypes.c_void_p or cty is ctypes.py_object:
        return coretypes.Pointer[coretypes.void]
    elif is_ctypes_array_type(cty):
        return arrayobject.Array[from_ctypes_type(cty._type_), cty._length_]
    elif is_ctypes_pointer_type(cty):
        return coretypes.Pointer[from_ctypes_type(cty._type_)]
    elif is_ctypes_struct_type(cty):
        fields = [(name, from_ctypes_type(field_type))
                      for name, field_type in cty._fields_]
        fields = fields or [('dummy', coretypes.int8)]
        return coretypes.struct_(fields)
    elif is_ctypes_function_type(cty):
        # from_ctypes_type(cty._restype_) # <- this value is arbitrary,
        # it's always a c_int
        restype = from_ctypes_type(ctypes_value.restype)
        argtypes = tuple(from_ctypes_type(argty) for argty in ctypes_value.argtypes)
        return coretypes.ForeignFunction[argtypes + (restype,)]
    else:
        raise NotImplementedError(cty)
