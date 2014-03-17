# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import math
import struct

import numpy as np

from flypy.types import *

if struct.pack('i', 1)[0] == '\1':
    nbo = '<' # little endian
else:
    nbo = '>' # big endian

def from_dtype(dtype):
    """
    Map a NumPy dtype to a minitype.

    >>> map_dtype(np.dtype(np.int32))
    int32
    >>> map_dtype(np.dtype(np.int64))
    int64
    >>> map_dtype(np.dtype(np.object))
    PyObject *
    >>> map_dtype(np.dtype(np.float64))
    float64
    >>> map_dtype(np.dtype(np.complex128))
    complex128
    """
    from flypy.types import (int8, int16, int32, int64,
                             uint8, uint16, uint32, uint64,
                             float32, float64, complex64, complex128,
                             struct_, object_)

    if dtype.byteorder not in ('=', nbo, '|') and dtype.kind in ('iufbc'):
        raise TypeError("Only native byteorder is supported", dtype)

    item_idx = int(math.log(dtype.itemsize, 2))
    if dtype.kind == 'i':
        return [int8, int16, int32, int64][item_idx]
    elif dtype.kind == 'u':
        return [uint8, uint16, uint32, uint64][item_idx]
    elif dtype.kind == 'f':
        if dtype.itemsize == 2:
            pass # half floats not supported yet
        elif dtype.itemsize == 4:
            return float32
        elif dtype.itemsize == 8:
            return float64
        elif dtype.itemsize == 16:
            raise TypeError("long double is not support")
    elif dtype.kind == 'b':
        return int8
    elif dtype.kind == 'c':
        if dtype.itemsize == 8:
            return complex64
        elif dtype.itemsize == 16:
            return complex128
        elif dtype.itemsize == 32:
            raise TypeError("long double is not support")
            # return complex256
    elif dtype.kind == 'V':
        fields = [(name, from_dtype(dtype.fields[name][0]))
                      for name in dtype.names]
        is_aligned = dtype.alignment != 1
        return struct_(fields, packed=not getattr(dtype, 'isalignedstruct',
                                                  is_aligned))
    elif dtype.kind == 'O':
        return object_

    # TODO:
    #elif dtype.kind == 'M':
    #    # Get datetime units from 2nd to last character in dtype string
    #    # Example dtype string: '<M8[D]', where D is datetime units
    #    return datetime(units=dtype.str[-2])
    #elif dtype.kind == 'm':
    #    # Get timedelta units from 2nd to last character in dtype string
    #    # Example dtype string: '<m8[D]', where D is timedelta units
    #    return timedelta(units=dtype.str[-2])


def to_dtype(type):
    from flypy.types import object_

    typemap = {
        int8     : np.int8,
        int16    : np.int16,
        int32    : np.int32,
        int64    : np.int64,
        uint8    : np.uint8,
        uint16   : np.uint16,
        uint32   : np.uint32,
        uint64   : np.uint64,

        float_   : np.float32,
        double   : np.float64,
        # longdouble: np.longdouble,

        short    : np.dtype('h'),
        int_     : np.dtype('i'),
        long_    : np.dtype('l'),
        longlong : np.longlong,
        ushort   : np.dtype('H'),
        uint     : np.dtype('I'),
        ulong    : np.dtype('L'),
        ulonglong: np.ulonglong,

        complex64: np.complex64,
        complex128: np.complex128,
        # complex256: getattr(np, 'complex256', None),

        bool_    : np.bool,
        object_  : np.object,
    }

    # TODO: records, datetime

    result = typemap[type]
    return np.dtype(result)