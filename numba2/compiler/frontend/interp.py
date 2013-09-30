# -*- coding: utf-8 -*-

"""
Interpreter for untyped bytecode
"""

from __future__ import print_function, division, absolute_import
from functools import partial

from numba2.compiler.special import lookup_operator
from pykit.ir import interp

def getfield(interp, obj, attr):
    if attr.startswith('__') and attr.startswith('__'):
        try:
            return partial(lookup_operator(attr), obj)
        except KeyError:
            pass

    if hasattr(obj, attr):
        return getattr(obj, attr)
    return getattr(type(obj), attr)

handlers = {
    'getfield': getfield,
}

def run(func, env=None, exc_model=None, args=()):
    env = env or {}
    env.setdefault('interp.handlers', {}).update(handlers)
    return interp.run(func, env, exc_model, args=args)