# -*- coding: utf-8 -*-

"""
Implements a runtime using CPython.
"""

# future feature absolute_import is not defined ?
from __future__ import print_function, division #, absolute_import

from cpython cimport *
from libc.stdint cimport uint64_t, int64_t
cimport numpy as npy

npy.import_array()

import numpy as np

import functools
try:
    import __builtin__ as builtin
except ImportError:
    import builtin

cdef extern from "Python.h":
    char *PyString_AS_STRING(object)

    ctypedef unsigned long Py_uintptr_t
    ctypedef struct PyTypeObject:
        pass


cdef extern from "numpy/arrayobject.h":
    PyObject *PyArray_NewFromDescr(PyTypeObject* subtype,
                                   PyObject* descr,
                                   int nd,
                                   npy.npy_intp* dims,
                                   npy.npy_intp* strides,
                                   void* data, int flags,
                                   PyObject* obj)

    PyTypeObject PyArray_Type

# ______________________________________________________________________
# Iterators

cdef public getiter(obj):
    return iter(obj)

cdef public next(obj):
    return next(obj)

# ______________________________________________________________________
# Attributes

cdef public getfield(obj, char *attr):
    return getattr(obj, attr)

cdef public void setfield(obj, char *attr, value):
    setattr(obj, attr, value)

# ______________________________________________________________________
# Indexing

cdef public getitem(obj, indices):
    return obj[indices]

cdef public void setitem(obj, indices, value):
    obj[indices] = value

cdef public slice slice(Py_ssize_t lower, Py_ssize_t upper, Py_ssize_t step):
    return builtin.slice(lower, upper, step)

# ______________________________________________________________________
# Basic operators

# Binary

cdef public add(x, y):
    return x + y

cdef public sub(x, y):
    return x - y

cdef public mul(x, y):
    return x * y

cdef public divide(x, y):
    return x / y

cdef public floordiv(x, y):
    return x // y

cdef public lshift(x, y):
    return x << y

cdef public rshift(x, y):
    return x >> y

cdef public bitor(x, y):
    return x | y

cdef public bitand(x, y):
    return x & y

# Unary

cdef public uadd(x):
    return +x

cdef public invert(x):
    return ~x

cdef public not_(x):
    return not x

cdef public usub(x):
    return -x

# Compare

cdef public bint lt(x, y):
    return x < y

cdef public bint le(x, y):
    return x <= y

cdef public bint gt(x, y):
    return x > y

cdef public bint ge(x, y):
    return x >= y

cdef public bint eq(x, y):
    return x == y

cdef public bint ne(x, y):
    return x != y

# ______________________________________________________________________
# Constructors

cdef public new_list(elems):
    return list(elems)

cdef public new_tuple(elems):
    return tuple(elems)

cdef public new_dict(keys, values):
    return dict(zip(keys, values))

cdef public new_set(elems):
    return set(elems)

cdef public new_string(elems):
    return bytes(elems)

cdef public new_unicode(elems):
    return unicode(elems)

cdef public new_complex(double real, double imag):
    return complex(real, imag)

cdef public call(obj, args):
    return obj(*args)

cdef public partial(func, values):
    return functools.partial(func, values)

# ______________________________________________________________________
# Primitives

cdef public print_(values):
    values = list(values)
    for value in values[:-1]:
        print(value,)
    print(values[-1])

# ______________________________________________________________________
# Pointers

cdef public fromvoidp(void *p):
    return <object> p

cdef public Py_uintptr_t address(PyObject *x):
    return <Py_uintptr_t> x

# ______________________________________________________________________
# Strings

cdef public tostring(obj):
    return str(obj)

cdef public torepr(obj):
    return repr(obj)

cdef public char * asstring(obj):
    assert isinstance(obj, str)
    return PyString_AS_STRING(obj)

cdef public fromstring(char *s, Py_ssize_t length):
    return s[:length]

# ______________________________________________________________________
# Special

cdef public bint istrue(obj):
    return bool(obj)

cdef public Py_ssize_t length(obj):
    return len(obj)

# ______________________________________________________________________

# NumPy

def create_array(Py_uintptr_t data, npy.npy_intp[:] shape,
                 npy.npy_intp[:] strides, dtype):
    """Create a dummy int8 array from the given data and size"""
    cdef npy.npy_intp *shape_p = &shape[0]
    cdef npy.npy_intp *strides_p = &strides[0]
    cdef int ndim = len(shape)

    assert len(shape) == len(strides)
    assert isinstance(dtype, np.dtype)

    Py_INCREF(dtype)
    cdef PyObject *result = PyArray_NewFromDescr(
        <PyTypeObject *> &PyArray_Type, <PyObject *> dtype, ndim,
        shape_p, strides_p, <void *> data, 0, NULL)

    return <object> result
# ______________________________________________________________________

cdef public void debug():
    print("debug...")
