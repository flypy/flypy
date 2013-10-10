# -*- coding: utf-8 -*-

"""
Dummy None implementation.
"""

from __future__ import print_function, division, absolute_import
import types
from ... import jit, typeof

@jit
class NoneType(object):
    layout = []


NoneValue = NoneType()


@typeof.case(types.NoneType)
def typeof(pyval):
     return NoneType