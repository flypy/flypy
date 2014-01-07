# -*- coding: utf-8 -*-

"""
Hacks that should go away when we have more powerful constructs.
"""

from __future__ import print_function, division, absolute_import
from flypy import jit
from flypy.runtime.obj.core import NoneType

# -- choice -- #

# NOTE: This works around the type inferencer not recognizing 'is not None'
#       as type checks.

# TODO: Integrate SCCP pass in type inferencer

@jit('a -> NoneType[] -> a')
def choose(a, b):
    return a

@jit('a -> b -> b')
def choose(a, b):
    return b