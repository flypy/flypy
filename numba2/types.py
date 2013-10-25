# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

__all__ = [
    'Function', 'Mono', 'Bool', 'Int', 'Float', 'Type', 'Constructor',
    'Pointer',
    'void', 'char', 'uchar', 'short', 'ushort',
    'int_', 'uint', 'long_', 'ulong', 'longlong', 'ulonglong',
    'size_t', 'npy_intp', 'bool_', 'string',
    'float32', 'float64','int8', 'int16', 'int32', 'int64', 'uint8',
    'uint16', 'uint32', 'uint64', 'Py_ssize_t',
    #'complex64', 'complex128', 'complex256', 'struct', 'Py_uintptr_t'
]


#===------------------------------------------------------------------===
# Types
#===------------------------------------------------------------------===
import struct

from blaze import dshape
from blaze.datashape import free, TypeVar, TypeConstructor
from blaze.datashape import Mono as Mono, Ellipsis as EllipsisT
from .runtime import Type, Constructor
from .runtime.obj import (Function, Pointer, Bool, Int, Float, Void, NoneType,
                          Tuple, StaticTuple, String, ForeignFunction)

#===------------------------------------------------------------------===
# Units
#===------------------------------------------------------------------===

string  = String[()]
bool_   = Bool[()]
void    = Void[()]
int8    = Int[8,  False]
int16   = Int[16, False]
int32   = Int[32, False]
int64   = Int[64, False]
uint8   = Int[8,  True]
uint16  = Int[16, True]
uint32  = Int[32, True]
uint64  = Int[64, True]
float32 = Float[32]
float64 = Float[64]

def signed(itemsize):
    return {1: int8, 2: int16, 4: int32, 8: int64}[itemsize]

def unsigned(itemsize):
    return {1: uint8, 2: uint16, 4: uint32, 8: uint64}[itemsize]

char      = int8
short     = signed(struct.calcsize('h'))
int_      = signed(struct.calcsize('i'))
long_     = signed(struct.calcsize('l'))
longlong  = signed(struct.calcsize('Q'))

uchar     = uint8
ushort    = unsigned(struct.calcsize('h'))
uint      = unsigned(struct.calcsize('i'))
ulong     = unsigned(struct.calcsize('l'))
ulonglong = unsigned(struct.calcsize('Q'))

# TODO: use the right platform sizes
double       = float64
Py_ssize_t   = int64
size_t       = uint64
Py_uintptr_t = uint64
npy_intp     = Py_ssize_t