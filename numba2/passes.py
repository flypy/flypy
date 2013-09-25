# -*- coding: utf-8 -*-

"""
Numba passes that perform translation, type inference, code generation, etc.
"""

from __future__ import print_function, division, absolute_import

from functools import partial, wraps

from .environment import root_env
from numba2.compiler.backend import preparation, backend
from .pipeline import run_pipeline
from .compiler.frontend import translate
from .compiler import simplification
from .compiler.typing import inference
from .compiler.typing.resolution import resolve_context, resolve_restype, rewrite_methods
from .prettyprint import dump, dump_cfg, dump_llvm, dump_optimized

from pykit.analysis import cfa, callgraph
from pykit.transform import dce
from pykit.codegen.llvm import verify, optimize

#===------------------------------------------------------------------===
# Passes
#===------------------------------------------------------------------===

frontend = [
    translate,
    dump_cfg,
    simplification,
    cfa,
]

typing = [
    inference,
    resolve_context,
    resolve_restype,
]

lower = [
    rewrite_methods,
]

optimize = [
    dce,
]

backend = [
    preparation,
    backend,
    verify,
    dump_llvm,
    optimize,
    dump_optimized,
]

passes = frontend + typing + lower + backend

#===------------------------------------------------------------------===
# Phases
#===------------------------------------------------------------------===

def cached(cache_name, passes):
    """Helper to perform caching for a phase"""
    def decorator(f):
        @wraps(f)
        def wrapper(func, env, passes=passes):
            cache = env[cache_name]
            if cache.lookup(func):
                return cache.lookup(func)

            new_func, new_env = f(func, env, passes)
            cache.insert(func, (new_func, new_env))
            return new_func, new_env

        return wrapper
    return decorator

def starcompose(f, g):
    """Helper to compose functions in a pipeline"""
    return lambda *args: f(*g(*args))

# ______________________________________________________________________
# Individual phases

@cached('numba.frontend.cache', frontend)
def translation_phase(func, env, passes):
    return run_pipeline(func, env, passes)

@cached('numba.typing.cache', typing + lower)
def typing_phase(func, env, passes):
    return run_pipeline(func, env, passes)

@cached('numba.opt.cache', optimize)
def optimization_phase(func, env, passes):
    return run_pipeline(func, env, passes)

@cached('numba.codegen.cache', backend)
def codegen_phase(func, env, passes):
    return run_pipeline(func, env, passes)

# ______________________________________________________________________
# Phase application

def apply_and_resolve(phase, func, env):
    """Apply a phase to a function and its dependences"""
    graph = env['numba.state.callgraph'] or callgraph.callgraph(func)
    for f in graph.node:
        phase(f, None)

dep_resolving = lambda phase: partial(apply_and_resolve, phase)

# ______________________________________________________________________
# Combined phases

class phase(object):
    # Just a little namespace to hold phases

    translation = translation_phase
    typing = dep_resolving(starcompose(typing_phase, translation_phase))
    opt = dep_resolving(starcompose(optimization_phase, typing))
    codegen = starcompose(codegen_phase, opt)