# -*- coding: utf-8 -*-

"""
Numba passes that perform translation, type inference, code generation, etc.
"""

from __future__ import print_function, division, absolute_import

from .environment import root_env
from numba2.compiler.backend import preparation, backend
from .pipeline import run_pipeline
from .compiler.frontend import translate
from .compiler import simplification
from .compiler.typing import inference
from .compiler.typing.resolution import resolve_context, resolve_restype, rewrite_methods
from .prettyprint import dump, dump_cfg, dump_llvm, dump_optimized

from pykit.analysis import cfa
from pykit.transform import dce
from pykit.codegen.llvm import verify, optimize

#===------------------------------------------------------------------===
# Passes
#===------------------------------------------------------------------===

frontend = [
    translate,
    dump_cfg,
]

typing = [
    simplification,
    cfa,
    inference,
]

resolution = [
    resolve_context,
    resolve_restype,
    rewrite_methods,
]

backend = [
    dce,
    preparation,
    backend,
    verify,
    dump_llvm,
    optimize,
    dump_optimized,
]

passes = frontend + typing + resolution + backend

#===------------------------------------------------------------------===
# Translation
#===------------------------------------------------------------------===

def translate(py_func, argtypes, restype=None, env=None, passes_=None):
    if env is None:
        env = dict(root_env)
    if passes_ is None:
        passes_ = passes

    # Types
    env['numba.typing.argtypes'] = argtypes
    env.setdefault('numba.typing.restype', restype)

    # State
    env['numba.state.py_func'] = py_func
    env['numba.state.func_globals'] = py_func.__globals__
    env['numba.state.func_code'] = py_func.__code__

    return run_pipeline(py_func, env, passes_)