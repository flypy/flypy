# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba2.typing import resolve

from blaze import overloading
from blaze.kernel import lookup_previous
from blaze.overloading import overload, Dispatcher
from blaze.util import flatargs


def overloadable(f):
    """
    Make a function overloadable, useful if there's no useful defaults to
    overload on
    """
    return Dispatcher()

# TODO: Cache results
def resolve_overloads(o, scope, bound):
    """
    Resolve the signatures of overloaded methods in the given scope. Further
    resolve any type variables listed in `bound` by their replacement.
    """
    result = Dispatcher()
    for (f, signature, kwds) in o.overloads:
        new_sig = resolve(signature, scope, bound)
        overload(new_sig, result, **kwds)(f)
    return result

def best_match(func_wrapper, argtypes):
    """
    Find the right overload for a numba function.

    Arguments
    ---------
    func_wrapper: FunctionWrapper
        The function

    argtypes: [Type]
        Types to call the overloaded function with

    Returns
    -------
    (py_func, result_signature)
    """
    o = func_wrapper.dispatcher
    scope = determine_scope(func_wrapper.py_func)
    bound = {} # TODO:
    overloaded = resolve_overloads(o, scope, bound)
    overload = overloading.best_match(overloaded, argtypes)
    return (overload.func, overload.resolved_sig)


def determine_scope(py_func):
    return py_func.__globals__