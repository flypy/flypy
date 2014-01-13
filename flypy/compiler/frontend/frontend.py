# -*- coding: utf-8 -*-

"""
Post-passes on untyped IR emitted by the front-end bytecode translation.
"""

from __future__ import print_function, division, absolute_import
import inspect

from .translation import Translate
from flypy.compiler.signature import compute_missing
from flypy.rules import typematch
from flypy.runtime.obj.core import make_tuple_type, StaticTuple, GenericTuple, extract_tuple_eltypes

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
    from flypy.compiler.signature import fill_missing_argtypes, flatargs

    argtypes = env["flypy.typing.argtypes"]

    if func.opaque:
        return argtypes

    # We make the simlifying assumption that `py_func` is the right overload,
    # which has not been determined yet. This means all overloads must have
    # the same types for defaults...
    py_func = func.py_func

    called_flags = env['flypy.state.called_flags']
    called_with_varargs = called_flags['varargs']
    called_with_keywords = called_flags['keywords']

    if called_with_varargs:
        argtypes = handle_unpacking_varargs(func, py_func, argtypes)
    else:
        # Fill out missing argtypes for defaults
        argtypes = fill_missing_argtypes(py_func, argtypes)

    # Handle varargs/keywords (*args, **kwargs)
    argtypes = flatargs(py_func, argtypes, {})
    result = list(argtypes)

    varargs, keywords = [], []
    if argtypes.have_keywords:
        #keywords = [result.pop()]
        raise TypeError("Keyword arguments are not yet supported")
    if argtypes.have_varargs:
        varargs = [make_tuple_type(result.pop())]

    argtypes = result + varargs + keywords
    #print(func, argtypes)
    return argtypes

def handle_unpacking_varargs(func, py_func, argtypes):
    """
    Handle unpacking of varargs:

        f(10, *args)
              ^^^^^

    We assume that unpacking consumes all positional arguments, i.e. we
    do not support shenanigans such as the following:

        def f(a, b=2, c=3):
            ...

        f(1, *(4,))

    To support that we'd have to call a wrapper function, that takes a tuple
    and supplies missing defaults dynamically.
    """
    argspec = inspect.getargspec(py_func)

    tuple_type = argtypes[-1]
    argtypes   = argtypes[:-1]
    missing  = compute_missing(py_func, argtypes)

    # Extract remaining argument types
    remaining = extract_tuple_eltypes(tuple_type, missing)
    if len(remaining) > missing and not argspec.varargs:
        raise TypeError(
            "Too many arguments supplied to function %s: %s" % (
                py_func, argtypes))

    return tuple(argtypes) + tuple(remaining)

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

    env['flypy.state.call_flags'] = t.call_annotations

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