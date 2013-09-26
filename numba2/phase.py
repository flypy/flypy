# -*- coding: utf-8 -*-

"""
Numba compiler phases (groupings of passes).
"""

from __future__ import print_function, division, absolute_import

from functools import partial, wraps

from .pipeline import run_pipeline
from .passes import frontend, typing, lower, optimizations, backend
from .compiler.overloading import best_match

from pykit.analysis import callgraph

#===------------------------------------------------------------------===
# Phases
#===------------------------------------------------------------------===

def cached(cache_name, passes, key=lambda func, env: func):
    """Helper to perform caching for a phase"""
    def decorator(f):
        @wraps(f)
        def wrapper(func, env, passes=passes):
            cache = env[cache_name]
            cache_key = key(func, env)
            if cache.lookup(cache_key):
                return cache.lookup(cache_key)

            new_func, new_env = f(func, env, passes)
            cache.insert(cache_key, (new_func, new_env))
            return new_func, new_env

        return wrapper
    return decorator

def starcompose(f, g):
    """Helper to compose functions in a pipeline"""
    return lambda *args: f(*g(*args))

# ______________________________________________________________________
# Individual phases

def _cache_key(func, env):
    return (func, tuple(env["numba.typing.argtypes"]))

def setup_phase(func, env):
    # Find implementation
    py_func, signature = best_match(func, env["numba.typing.argtypes"])

    # Update environment
    env["numba.state.function_wrapper"] = func
    env['numba.state.opaque'] = func.opaque
    env["numba.typing.restype"] = signature.restype
    env["numba.typing.argtypes"] = signature.argtypes

    return py_func, env

@cached('numba.frontend.cache', frontend)
def translation_phase(func, env, passes):
    if env["numba.state.opaque"]:
        return func, env
    return run_pipeline(func, env, passes)

@cached('numba.typing.cache', typing + lower, key=_cache_key)
def typing_phase(func, env, passes):
    return run_pipeline(func, env, passes)

@cached('numba.opt.cache', optimizations)
def optimization_phase(func, env, passes):
    apply_and_resolve(partial(run_pipeline, passes=passes), func, env)
    return func, env

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

setup = setup_phase
translation = starcompose(translation_phase, setup)
typing = starcompose(typing_phase, translation)
opt = starcompose(optimization_phase, typing)
codegen = starcompose(codegen_phase, opt)