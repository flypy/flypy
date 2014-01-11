# -*- coding: utf-8 -*-

"""
Post-passes on untyped IR emitted by the front-end bytecode translation.
"""

from __future__ import print_function, division, absolute_import
import inspect

from .translation import Translate
from flypy.runtime.obj.core import make_tuple_type

from pykit.ir import Builder

#===------------------------------------------------------------------===
# Entrypoint
#===------------------------------------------------------------------===

def setup(func, env):
    """
    Set up function's environment. Very first pass that runs in the pipeline.
    """
    from flypy.compiler.overloading import best_match

    # -------------------------------------------------
    # Find Python function implementation

    argtypes = simplify_argtypes(func, env)
    py_func, signature, kwds = best_match(func, list(argtypes))

    # -------------------------------------------------
    # Update environment
    env["flypy.state.func_name"] = py_func.__name__
    env["flypy.state.function_wrapper"] = func
    env["flypy.state.opaque"] = func.opaque
    env["flypy.typing.restype"] = signature.restype
    env["flypy.typing.argtypes"] = signature.argtypes
    env["flypy.state.crnt_func"] = func
    env["flypy.state.options"] = dict(kwds)
    env["flypy.state.copies"] = {}

    if kwds.get("infer_restype"):
        env["flypy.typing.restype"] = kwds["infer_restype"](argtypes)

    return py_func, env

def simplify_argtypes(func, env):
    """
    Simplify the argtypes for non-opaque functions:

        *args    -> tuple
        **kwargs -> dict
    """
    from flypy.compiler.overloading import flatargs

    argtypes = env["flypy.typing.argtypes"]

    if func.opaque:
        return argtypes

    # Simlifying assumption...
    py_func = func.py_func
    argtypes = list(flatargs(py_func, argtypes, {}))

    varargs, keywords = [], []
    if argtypes and isinstance(argtypes[-1], dict):
        #keywords = [argtypes.pop()]
        raise TypeError("Keyword arguments are not yet supported")
    if argtypes and isinstance(argtypes[-1], tuple):
        varargs = [make_tuple_type(argtypes.pop())]

    return argtypes + varargs + keywords


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
