# -*- coding: utf-8 -*-

"""
Post-passes on untyped IR emitted by the front-end bytecode translation.
"""

from __future__ import print_function, division, absolute_import
from .translation import Translate

from pykit.ir import Builder

#===------------------------------------------------------------------===
# Entrypoint
#===------------------------------------------------------------------===

def setup(func, env):
    from numba2.compiler.overloading import best_match

    # -------------------------------------------------
    # Find Python function implementation

    argtypes = env["numba.typing.argtypes"]
    py_func, signature, kwds = best_match(func, list(argtypes))

    # -------------------------------------------------
    # Update environment
    env["numba.state.func_name"] = py_func.__name__
    env["numba.state.function_wrapper"] = func
    env["numba.state.opaque"] = func.opaque
    env["numba.typing.restype"] = signature.restype
    env["numba.typing.argtypes"] = signature.argtypes
    env["numba.state.crnt_func"] = func
    env["numba.state.options"] = dict(kwds)
    env["numba.state.copies"] = {}

    if kwds.get("infer_restype"):
        env["numba.typing.restype"] = kwds["infer_restype"](argtypes)

    return py_func, env

def translate(py_func, env):
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

    t = Translate(py_func, env)
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
            b.splitblock(terminate=True, preserve_exc=False)
            op.delete()
