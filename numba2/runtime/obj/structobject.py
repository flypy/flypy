# -*- coding: utf-8 -*-

"""
Struct object implementation.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from ... import jit, typeof

def struct_(fields, name=None):
    if (not isinstance(fields, list) or not fields or not
            isinstance(fields[0], tuple) or not len(fields[0]) == 2):
        raise TypeError("Excepted a list of two-tuples")

    @jit('Struct[a, b]')
    class Struct(object):
        layout = list(fields)

    if name:
        Struct.__name__ = name

    return Struct