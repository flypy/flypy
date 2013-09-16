# -*- coding: utf-8 -*-

"""
Post-passes on untyped IR emitted by the front-end bytecode translation.
"""

from __future__ import print_function, division, absolute_import

import __builtin__
import inspect
import dis
import operator
import collections

from numba2.errors import error_context, CompileError, EmptyStackError
from numba2.compiler import typeof
from .bytecode import ByteCode

from pykit.ir import Function, Builder, OpBuilder, Op, Const, ops, defs
from pykit import types

#===------------------------------------------------------------------===
# Exceptions
#===------------------------------------------------------------------===

def simplify_exceptions(func, env=None):
    """
    Rewrite exceptions emitted by the front-end:

        exc_end -> split block
    """
    # for block in func.blocks:
    #     for op in block.leaders:
    #         if op.opcode == 'exc_'
    b = Builder(func)
    for op in func.ops:
        if op.opcode == 'exc_end':
            b.position_after(op)
            b.splitblock(terminate=True)
            op.delete()