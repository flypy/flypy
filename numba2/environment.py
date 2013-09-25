# -*- coding: utf-8 -*-

"""
Numba compilation environment.
"""

from __future__ import print_function, division, absolute_import

from .utils import FrozenDict
from .caching import Cache, InferenceCache

root_env = FrozenDict({
    # Command line args
    'numba.cmdopts':        {},

    # Caching
    'numba.frontend.cache':     Cache(),
    'numba.typing.cache':       Cache(),
    'numba.inference.cache':    InferenceCache(),
    'numba.opt.cache':          Cache(),
    'numba.codegen.cache':      Cache(),

    # General state
    'numba.state.py_func':      None,   # This value may be None
    'numba.state.func_globals': None,
    'numba.state.func_code':    None,
    'numba.state.callgraph':    None,

    # Typing
    'numba.typing.restype': None,       # Input/Output
    'numba.typing.argtypes': None,      # Input
    'numba.typing.signature': None,     # Output
    'numba.typing.context': None,       # Output
    'numba.typing.constraints': None,   # Output

    # Flags
    'numba.verify':         True,
    'numba.optimize':       True,

    # Codegen
    "codegen.llvm.opt":     None,
    "codegen.llvm.engine":  None,
    "codegen.llvm.module":  None,
    "codegen.llvm.machine": None,
    "codegen.llvm.ctypes":  None,
})

#===------------------------------------------------------------------===
# New envs
#===------------------------------------------------------------------===

def fresh_env(py_func, argtypes, restype=None, env=None):
    """
    Allocate a new environment, optionally from a given environment.
    """
    if env is None:
        env = root_env

    env = dict(env)

    # Types
    env['numba.typing.argtypes'] = argtypes
    env.setdefault('numba.typing.restype', restype)

    # State
    env['numba.state.py_func'] = py_func
    env['numba.state.func_globals'] = py_func.__globals__
    env['numba.state.func_code'] = py_func.__code__

    return env