# -*- coding: utf-8 -*-

"""
Copy functions to a function with a new name.
"""

from __future__ import print_function, division, absolute_import

from pykit.ir import copy_function
from pykit.utils import make_temper

temper = make_temper()

def copying(func, env):
    #new_func = copy_function(func)
    new_func = func
    new_func.name = temper(new_func.name)
    return new_func, env

run = copying