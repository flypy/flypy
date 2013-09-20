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

def translate(func, env=None):
    """
    Entry point.

    Parameters
    ----------

    func : Python function
        Python function to translate

    Returns : pykit.ir.Function
        Untyped pykit function. All types are Opaque unless they are constant.
    """
    if env:
        cache = env['numba.frontend.cache']
        if cache.lookup(func):
            return cache.lookup(func)

    # -------------------------------------------------
    # Translate

    t = Translate(func)
    t.initialize()
    t.interpret()

    if env:
        cache.insert(func, t.dst)

    func = t.dst

    # -------------------------------------------------
    # Postpasses

    simplify_exceptions(func, env)
    rewrite_calls(func, env)

    return func

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

#===------------------------------------------------------------------===
# Calls
#===------------------------------------------------------------------===

def rewrite_calls(func, env):
    """
    Rewrite 'pycall' instructions to 'call'.

    This recurses into the front-end translator, so this supports recursion
    since the function is already in the cache.
    """
    b = OpBuilder()
    for op in func.ops:
        if op.opcode == 'pycall':
            func, args = op.args[0].const, op.args[1:]
            if isinstance(func, FunctionWrapper):
                translated = translate(func.py_func, env)
                newop = b.call(types.Opaque, [translated, args], op.result)
                op.replace(newop)