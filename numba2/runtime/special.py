# -*- coding: utf-8 -*-

"""
Special numba functions.
"""

from __future__ import print_function, division, absolute_import
from numba2 import jit

__all__ = ['typeof']

@jit('a -> Type[a]')
def typeof(obj):
    raise NotImplementedError("Not implemented at the python level")
