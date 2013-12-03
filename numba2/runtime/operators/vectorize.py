# -*- coding: utf-8 -*-

"""
numba.vectorize().
"""

from __future__ import print_function, division, absolute_import
from itertools import starmap

from numba2 import sjit, jit, typeof, conversion, parse

def vectorize(py_func, signatures, **kwds):
    signature = parse(signatures[0])
    nargs = len(signature.argtypes)
    py_func = make_ndmap(py_func, nargs)

    func = jit(py_func, signatures[0], **kwds)
    for signature in signatures[1:]:
        func.overload(py_func, signature, **kwds)

def make_ndmap(py_func, nargs):
    raise NotImplementedError("This will be messy")