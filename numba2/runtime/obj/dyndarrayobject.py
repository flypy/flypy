# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from numba2 import jit, sjit, types
from dynd import nd, ndt, _lowlevel
from numba2.runtime.casting import cast
from numba2.support import dynd_support

@sjit('MemoryBlockData[]')
class MemoryBlockData(object):
    layout = [
        ('atomic_refcount', 'int32'),
        ('type', 'uint32')
    ]

@sjit('DyNDArrayLayout[a]')
class DyNDArrayLayout(object):
    layout = [
        ('mbd', 'MemoryBlockData'),
        ('type', 'Pointer[void]'),
        ('data_pointer', 'Pointer[void]'),
        ('flags', 'uint64'),
        ('data_reference', 'Pointer[MemoryBlockData]'),
        ('meta', 'a')
    ]

@sjit('VarDim[a]')
class VarDim(object):
    layout = [
        ('blockref', 'Pointer[MemoryBlockData]'),
        ('stride', 'int64'), # 'intptr'),
        ('offset', 'int64'), # 'intptr'),
    ]

@sjit('StridedDim[a]')
class StridedDim(object):
    layout = [
        ('size', 'int64'), # 'intptr')
        ('stride', 'int64'), # 'intptr'),
    ]

@sjit('FixedDim[size, stride, a]')
class FixedDim(object):
    layout = []

@sjit('WrapNumbaType[a]')
class WrapNumbaType(object):
    layout = []

@jit('DyNDArray[a]')
class DyNDArray(object):
    """
    DyND Array Object.
    """

    layout = [
        ('arr', 'Pointer[DyNDArrayLayout[a]]')
    ]

    @jit('DyNDArray[a] -> Pointer[DyNDArrayLayout[a]] -> void')
    def __init__(self, arr_ptr):
        self.arr = arr_ptr
        _lowlevel.memory_block_incref(self.arr)

    @jit('DyNDArray[a] -> void')
    def __del__(self):
        _lowlevel.memory_block_decref(self.arr)

def _meta_type(tp):
    """
    Builds a numba type for the metadata of the
    provided dynd type object.
    """
    tid = tp.type_id
    if tid == 'var_dim':
        return VarDim[_meta_type(tp.element_type)]
    elif tid == 'strided_dim':
        return StridedDim[_meta_type(tp.element_type)]
    elif tid == 'fixed_dim':
        return FixedDim[tp.fixed_dim_size, tp.fixed_dim_stride, _meta_type(tp.element_type)]
    else:
        ntp = dynd_support.from_dynd_type(tp)
        return WrapNumbaType[ntp]

def fromdynd(arr):
    """Build a DyNDArray from an nd.array object"""

    # Construct the DyNDArrayLayout[a] pointer type
    ntp = DyNDArrayLayout[_meta_type(nd.type_of(arr))]
    # Get the raw array pointer
    arr_raw_ptr = _lowlevel.get_array_ptr(arr)
    arr_ptr = cast(arr_raw_ptr, types.Pointer[ntp])
    # Return a wrapping DyNDArray
    return DyNDArray(arr_ptr)

