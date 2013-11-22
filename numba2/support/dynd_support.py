# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from dynd import nd, ndt
import ctypes
from .. import types

_from_dynd_typemap = {
    'bool' : types.bool_,
    'int8' : types.int8,
    'int16' : types.int16,
    'int32' : types.int32,
    'int64' : types.int64,
    'uint8' : types.uint8,
    'uint16' : types.uint16,
    'uint32' : types.uint32,
    'uint64' : types.uint64,
    'float32' : types.float32,
    'float64' : types.float64,
    'cfloat32' : types.complex64,
    'cfloat64' : types.complex128,
}

def from_dynd_type(tp):
    """
    Map a DyND type to a Numba type.
    """
    if not isinstance(tp, ndt.type):
        raise TypeError('Expecting a dynd ndt.type, not %s' % type(tp))

    res = _from_dynd_typemap.get(str(tp), None)
    if res is not None:
        return res
    if tp.type_id == 'cstruct':
        fields = [(str(fname), from_dtype_type(ndt.type(ftp)))
                    for fname, ftp in zip(tp.field_names, tp.field_types)]
        return types.struct(fields, packed=False)

    raise TypeError('Unsupported dynd type %s' % tp)

_to_dynd_typemap = {
    types.bool_ : ndt.bool,
    types.int8 : ndt.int8,
    types.int16 : ndt.int16,
    types.int32 : ndt.int32,
    types.int64 : ndt.int64,
    types.uint8 : ndt.uint8,
    types.uint16 : ndt.uint16,
    types.uint32 : ndt.uint32,
    types.uint64 : ndt.uint64,
    types.float_ : ndt.float32,
    types.double : ndt.float64,
    types.complex64 : ndt.cfloat32,
    types.complex128 : ndt.cfloat64,
    types.short : ndt.type(ctypes.c_short),
    types.int_ : ndt.type(ctypes.c_int),
    types.long_ : ndt.type(ctypes.c_long),
    types.longlong : ndt.type(ctypes.c_longlong),
    types.ushort : ndt.type(ctypes.c_ushort),
    types.uint : ndt.type(ctypes.c_uint),
    types.ulong : ndt.type(ctypes.c_ulong),
    types.ulonglong : ndt.type(ctypes.c_ulonglong),
}

def to_dynd_type(ntp):
    """
    Map a Numba type to a DyND type.
    """
    return _to_dynd_typemap[ntp]

