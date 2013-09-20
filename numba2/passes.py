# -*- coding: utf-8 -*-

"""
Numba passes that perform translation, type inference, code generation, etc.
"""

from __future__ import print_function, division, absolute_import

from .environment import root_env
from .pipeline import run_pipeline
from .frontend import translate
from .compiler import simplification, inference
from .backend import backend

#===------------------------------------------------------------------===
# Utils
#===------------------------------------------------------------------===

def dump(func, env):
    print(func)

#===------------------------------------------------------------------===
# Passes
#===------------------------------------------------------------------===

passes = [
    translate,
    simplification,
    inference,
    backend,
]

#===------------------------------------------------------------------===
# Translation
#===------------------------------------------------------------------===

def translate(py_func, argtypes, restype=None, env=None, passes=passes):
    if env is None:
        env = dict(root_env)

    # Types
    env['numba.typing.argtypes'] = argtypes
    env.setdefault('numba.typing.restype', restype)

    # State
    env['numba.state.py_func'] = py_func
    env['numba.state.func_globals'] = py_func.__globals__
    env['numba.state.func_code'] = py_func.__code__

    run_pipeline(py_func, env, passes)