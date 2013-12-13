# -*- coding: utf-8 -*-

"""
Numba compiler phases (groupings of passes).
"""

from __future__ import print_function, division, absolute_import

import os
from functools import partial, wraps

from numba2.compiler.overloading import best_match
from numba2.viz.viz import dump
from .pipeline import run_pipeline
from . import passes
from .environment import fresh_env

from pykit.analysis import callgraph

#===------------------------------------------------------------------===
# Phases
#===------------------------------------------------------------------===

def cached(phase_name, key=lambda func, env: func):
    """Helper to perform caching for a phase"""
    cache_name = '.'.join([phase_name, 'cache'])

    def decorator(f):
        @wraps(f)
        def wrapper(func, env, *args):
            cache = env[cache_name]
            cache_key = key(func, env)

            # -------------------------------------------------
            # Check cache

            env['numba.state.phase'] = phase_name

            if cache.lookup(cache_key):
                new_func, new_env = cache.lookup(cache_key)
                if phase_name == 'numba.frontend':
                    # TODO: This is a hack! Do manual caching in
                    # translation_phase below!
                    new_env = env # Don't lose the argtypes
                return new_func, new_env

            # -------------------------------------------------
            # Apply phase & cache

            new_func, new_env = f(func, env, *args)
            cache.insert(cache_key, (new_func, new_env))

            return new_func, new_env

        return wrapper
    return decorator

def starcompose(f, g):
    """Helper to compose functions in a pipeline"""
    return lambda *args: f(*g(*args))

# ______________________________________________________________________
# Utils

def _cache_key(func, env):
    return (func, tuple(env["numba.typing.argtypes"]))

def _deps(func, debug=False):
    graph = callgraph.callgraph(func)
    if debug:
        import networkx as nx
        G = nx.DiGraph()
        for src in graph.node:
            for dst in graph.neighbors(src):
                G.add_edge(src.name, dst.name)
        dump(G, os.path.expanduser("~/callgraph.dot"))

    return graph.node

# ______________________________________________________________________
# Individual phases

def setup_phase(func, env):
    # -------------------------------------------------
    # Find Python function implementation
    argtypes = env["numba.typing.argtypes"]
    py_func, signature, kwds = best_match(func, list(argtypes))

    # -------------------------------------------------
    # Update environment
    env["numba.state.func_name"] = py_func.__name__
    env["numba.state.function_wrapper"] = func
    env["numba.state.opaque"] = func.opaque
    env["numba.typing.restype"] = signature.restype
    env["numba.typing.argtypes"] = signature.argtypes
    env["numba.state.crnt_func"] = func
    env["numba.state.options"] = dict(kwds)
    env["numba.state.copies"] = {}
    env["numba.state.phase"] = "setup"

    if kwds.get("infer_restype"):
        env["numba.typing.restype"] = kwds["infer_restype"](argtypes)

    return py_func, env

@cached('numba.frontend')
def translation_phase(func, env, passes=passes.frontend):
    if env["numba.state.opaque"]:
        return func, env
    new_func, new_env = run_pipeline(func, env, passes)
    envs = env['numba.state.envs']
    envs[new_func] = new_env
    return new_func, new_env

@cached('numba.typing', key=_cache_key)
def typing_phase(func, env, passes=passes.typing):
    return run_pipeline(func, env, passes)

@cached('numba.generators', key=_cache_key)
def generator_phase(func, env, passes=passes.generators, dependences=None):
    if dependences is None:
        dependences = _deps(func)
    envs = env["numba.state.envs"]
    for f in dependences:
        if f != func:
            generator_phase(f, envs[f], passes, [])
    return run_pipeline(func, env, passes)

@cached('numba.lowering', key=_cache_key)
def lowering_phase(func, env, passes=passes.lowering, dependences=None):
    if dependences is None:
        dependences = _deps(func)
    envs = env["numba.state.envs"]
    for f in dependences:
        if f != func:
            lowering_phase(f, envs[f], passes, [])
    return run_pipeline(func, env, passes)

@cached('numba.opt')
def optimization_phase(func, env, passes=passes.optimizations, dependences=None):
    envs = env["numba.state.envs"]
    if dependences is None:
        dependences = _deps(func)

    for f in dependences:
        if f != func:
            optimization_phase(f, envs[f], passes, [])
    run_pipeline(func, env, passes)

    return func, env

@cached('numba.ll_lowering')
def ll_lowering_phase(func, env, passes=passes.ll_lowering, dependences=None):
    envs = env["numba.state.envs"]
    if dependences is None:
        dependences = _deps(func)

    # Lower all dependences
    for f in dependences:
        if f != func:
            ll_lowering_phase(f, envs[f], passes, [])

    # Lower function
    run_pipeline(func, env, passes)

    return func, env

def codegen_phase(func, env):
    cache = env['numba.codegen.cache']
    envs = env["numba.state.envs"]

    if func in cache:
        return cache[func]

    dependences = [d for d in _deps(func, debug=True) if d not in cache]

    for f in dependences:
        run_pipeline(f, envs[f], passes.backend_init)
    for f in dependences:
        run_pipeline(f, envs[f], passes.backend_run)
    for f in dependences:
        e = envs[f]
        lfunc = e["numba.state.llvm_func"]
        run_pipeline(lfunc, envs[f], passes.backend_finalize)
        cache.insert(f, (lfunc, e))

    return env["numba.state.llvm_func"], env

# ______________________________________________________________________
# Phase application

def apply_and_resolve(phase, func, env, graph=None):
    """Apply a phase to a function and its dependences"""
    graph = graph or callgraph.callgraph(func)
    envs = env["numba.state.envs"]
    for f in graph.node:
        phase(f, envs[f])

# ______________________________________________________________________
# Combined phases

class CompileError(BaseException):
    """Compile error"""

def phasecompose(phase1, phase2):
    @wraps(phase1)
    def wrapper(func, env):
        try:
            return phase1(*phase2(func, env))
        except Exception, e:
            #print("-----------------------------")
            #print("Exception occurred when compiling %s with argtypes %s" % (
            #                              func, env["numba.typing.argtypes"]))
            raise #CompileError(str(e))

    return wrapper

setup = setup_phase
translation = phasecompose(translation_phase, setup)
typing = phasecompose(typing_phase, translation)
generators = phasecompose(generator_phase, typing)
lower = phasecompose(lowering_phase, generators)
opt = phasecompose(optimization_phase, lower)
ll_lower = phasecompose(ll_lowering_phase, opt)
codegen = phasecompose(codegen_phase, ll_lower)

# ______________________________________________________________________
# Naming

phases = {
    "setup":        setup,
    "translation":  translation,
    "typing":       typing,
    "generators":   generators,
    "lower":        lower,
    "opt":          opt,
    "ll_lower":     ll_lower,
    "codegen":      codegen,
}

# ______________________________________________________________________
# Apply

def apply_phase(phase, nb_func, argtypes):
    env = fresh_env(nb_func, argtypes)
    return phase(nb_func, env)