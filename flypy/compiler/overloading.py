# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import inspect

from flypy.typing import resolve, to_blaze

from datashape import overloading
from datashape import coretypes as T
from datashape.overloading import lookup_previous, flatargs as simple_flatargs
from datashape.overloading import overload, Dispatcher
from datashape.util import gensym

def overloadable(f):
    """
    Make a function overloadable, useful if there's no useful defaults to
    overload on
    """
    return Dispatcher()

def best_match(func_wrapper, argtypes):
    """
    Find the right overload for a flypy function.

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
    overloaded = func_wrapper.resolve_dispatcher()
    argtypes = [to_blaze(t) for t in argtypes]

    overload = overloading.best_match(overloaded, argtypes)
    scope = determine_scope(overload.func)
    signature = resolve(overload.resolved_sig, scope, {})
    return (overload.func, signature, overload.kwds)


def determine_scope(py_func):
    return py_func.__globals__