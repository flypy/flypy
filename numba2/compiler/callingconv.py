# -*- coding: utf-8 -*-

"""
Numba calling convention.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from pykit.ir import Op, Const, FuncArg, Function

def is_numba_cc(op):
    return isinstance(op, (Function, FuncArg, Op))

# TODO: Move byref and stack_allocate here from runtime/conversion.py