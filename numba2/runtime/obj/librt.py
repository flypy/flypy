# -*- coding: utf-8 -*-

"""
Implement objects.
"""

from __future__ import print_function, division, absolute_import

import os
from os.path import abspath, dirname, join

from . import Void, Pointer
from . import libcpy # initialize

import ctypes
#import cffi

#===------------------------------------------------------------------===
# Setup
#===------------------------------------------------------------------===

dir = dirname(abspath(__file__))
exts = (".so", ".dylib", ".dll")
files = [x for x in os.listdir(dir)
               if x.startswith("libcpy") and x.endswith(exts)]
if not files:
    raise OSError("No compiled library found, try running setup.py")

[fname] = files
lib = ctypes.PyDLL(join(dir, fname))

#===------------------------------------------------------------------===
# Declarations
#===------------------------------------------------------------------===

#typedef struct _object {
#    Py_ssize_t ob_refcnt;
#    void *ob_type;
#} PyObject;

def declare_in(lib, name, restype, argtypes):
    f = getattr(lib, name)
    f.restype = restype
    f.argtypes = argtypes
    return f


def declare(name, restype, argtypes):
    return declare_in(lib, name, restype, argtypes)


obj = ctypes.py_object

Py_IncRef  = declare_in(ctypes.pythonapi, 'Py_IncRef', None, [obj])
Py_DecRef  = declare_in(ctypes.pythonapi, 'Py_DecRef', None, [obj])

getiter    = declare('getiter' , obj, [obj])
next       = declare('next'    , obj, [obj])

getfield   = declare('getfield', obj, [obj, obj])
setfield   = declare('setfield', obj, [obj, obj])
getitem    = declare('getitem' , obj, [obj, obj])
setitem    = declare('setitem' , obj, [obj, obj])

add        = declare('add'     , ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p])
sub        = declare('sub'     , obj, [obj, obj])
mul        = declare('mul'     , obj, [obj, obj])
divide     = declare('divide'  , obj, [obj, obj])
floordiv   = declare('floordiv', obj, [obj, obj])
lshift     = declare('lshift'  , obj, [obj, obj])
rshift     = declare('rshift'  , obj, [obj, obj])
bitor      = declare('bitor'   , obj, [obj, obj])
bitand     = declare('bitand'  , obj, [obj, obj])

lt         = declare('lt'      , ctypes.c_int, [obj, obj])
le         = declare('le'      , ctypes.c_int, [obj, obj])
gt         = declare('gt'      , ctypes.c_int, [obj, obj])
ge         = declare('ge'      , ctypes.c_int, [obj, obj])
eq         = declare('eq'      , ctypes.c_int, [obj, obj])
ne         = declare('ne'      , ctypes.c_int, [obj, obj])

uadd       = declare('uadd'    , obj, [obj])
invert     = declare('invert'  , obj, [obj])
not_       = declare('not_'    , obj, [obj])
usub       = declare('usub'    , obj, [obj])

fromvoidp       = declare('fromvoidp' , obj, [ctypes.c_void_p])
address         = declare('address', ctypes.c_uint64, [ctypes.py_object])
istrue          = declare('istrue'       , ctypes.c_int, [obj])
tostring        = declare('tostring'     , obj, [obj])
torepr          = declare('torepr'       , obj, [obj])
asstring        = declare('asstring'     , ctypes.POINTER(ctypes.c_int8), [obj])
fromstring      = declare('fromstring'   , obj, [ctypes.POINTER(ctypes.c_int8),
                                                 ctypes.c_ssize_t])

# CFFI always releases the GIL...
#ffi.cdef("""
#typedef long Py_ssize_t;
#typedef void PyObject;
#
#void Py_IncRef(PyObject *);
#void Py_DecRef(PyObject *);
#
#PyObject *getiter(PyObject *);
#PyObject *next(PyObject *);
#
#PyObject *getfield(PyObject *, PyObject *);
#PyObject *setfield(PyObject *, PyObject *, PyObject *);
#
#PyObject *getitem(PyObject *, PyObject *);
#PyObject *setitem(PyObject *, PyObject *);
#
#PyObject *add(PyObject *, PyObject *);
#PyObject *sub(PyObject *, PyObject *);
#PyObject *mul(PyObject *, PyObject *);
#PyObject *divide(PyObject *, PyObject *);
#PyObject *floordiv(PyObject *, PyObject *);
#PyObject *lshift(PyObject *, PyObject *);
#PyObject *rshift(PyObject *, PyObject *);
#PyObject *bitor(PyObject *, PyObject *);
#PyObject *bitand(PyObject *, PyObject *);
#
#PyObject *lt(PyObject *, PyObject *);
#PyObject *le(PyObject *, PyObject *);
#PyObject *gt(PyObject *, PyObject *);
#PyObject *ge(PyObject *, PyObject *);
#PyObject *eq(PyObject *, PyObject *);
#PyObject *ne(PyObject *, PyObject *);
#
#PyObject *uadd(PyObject *);
#PyObject *invert(PyObject *);
#PyObject *not_(PyObject *);
#PyObject *usub(PyObject *);
#
#int istrue(PyObject *);
#char *asstring(PyObject *);
#PyObject *fromstring(char *, Py_ssize_t);
#""")

#lib = ffi.dlopen(join(dir, fname))

#PyObject_p = typeof(lib.add).parameters[0]
PyObject_p = Pointer[Void[()]]
