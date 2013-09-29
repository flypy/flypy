# -*- coding: utf-8 -*-

"""
Number interfaces.
"""

from __future__ import print_function, division, absolute_import
from numba2 import jit

__all__ = ['Type']

@jit('Type[T]')
class Type(object):
    layout = []
