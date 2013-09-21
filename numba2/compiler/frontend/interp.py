# -*- coding: utf-8 -*-

"""
Interpreter for untyped bytecode
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import interp

def pycall(interp, func, *args):
    return func(*args)

def getfield(interp, obj, attr):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    return getattr(type(obj), attr)

handlers = {
    'pycall':   pycall,
    'getfield': getfield,
}

def run(func, env=None, exc_model=None, args=()):
    env = env or {}
    env.setdefault('interp.handlers', {}).update(handlers)
    return interp.run(func, env, exc_model, args=args)