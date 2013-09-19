# -*- coding: utf-8 -*-

"""
Function implementation.
"""

from __future__ import print_function, division, absolute_import

from ... import jit
from blaze.datashape import Function as FunctionType

@jit(FunctionType())
class Function(object):
    layout = []