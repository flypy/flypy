# -*- coding: utf-8 -*-

"""
Post-passes on untyped IR emitted by the front-end bytecode translation.
"""

from __future__ import print_function, division, absolute_import

from numba2.functionwrapper import FunctionWrapper
from .translation import Translate

from pykit import types
from pykit.ir import Builder, OpBuilder

#===------------------------------------------------------------------===
# Entrypoint
#===------------------------------------------------------------------===

def translate(py_func, env=None):
    """
    Entry point.

    Parameters
    ----------

    func : Python function
        Python function to translate

    Returns : pykit.ir.Function
        Untyped pykit function. All types are Opaque unless they are constant.
    """
    # -------------------------------------------------
    # Translate

    t = Translate(py_func)
    t.initialize()
    t.interpret()
    func = t.dst

    return func, env

#===------------------------------------------------------------------===
# Exceptions
#===------------------------------------------------------------------===

def simplify_exceptions(func, env=None):
    """
    Rewrite exceptions emitted by the front-end:

        exc_end -> split block
    """
    b = Builder(func)
    for op in func.ops:
        if op.opcode == 'exc_end':
            b.position_after(op)
            b.splitblock(terminate=True)
            op.delete()
