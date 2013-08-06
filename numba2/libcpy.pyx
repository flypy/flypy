# -*- coding: utf-8 -*-

"""
Implements a runtime using CPython.
"""

from __future__ import print_function, division, absolute_import

from cpython cimport *

import functools
try:
    import __builtin__ as builtin
except ImportError:
    import builtin

# ______________________________________________________________________
# Iterators

cdef public getiter(obj):
    return iter(obj)

cdef public next(obj):
    return next(obj)

# ______________________________________________________________________
# Attributes

cdef public getfield(obj, str attr):
    return getattr(obj, attr)

cdef public void setfield(obj, str attr, value):
    setattr(obj, attr, value)

# ______________________________________________________________________
# Indexing

cdef public getindex(obj, indices):
    return obj[indices]

cdef public void setindex(obj, indices, value):
    obj[indices] = value

cdef public getslice(obj, indices):
    return obj[indices]

cdef public void setslice(obj, indices, value):
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

cdef public div(x, y):
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

cdef public lt(x, y):
    return x < y

cdef public lte(x, y):
    return x <= y

cdef public gt(x, y):
    return x > y

cdef public gte(x, y):
    return x >= y

cdef public eq(x, y):
    return x == y

cdef public noteq(x, y):
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