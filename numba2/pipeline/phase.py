# -*- coding: utf-8 -*-

"""
Numba compiler phases (groupings of passes).
"""

from __future__ import print_function, division, absolute_import

from functools import partial, wraps

from numba2.compiler.overloading import best_match
from .pipeline import run_pipeline
from .passes import (frontend, typing, optimizations, prelowering, lowering,
                     backend_init, backend_run, backend_finalize,
                     dpp_backend_run, dpp_backend_finalize)
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

#def _deps(func):
#    return callgraph.callgraph(func).node

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
def translation_phase(func, env, passes=frontend):
    if env["numba.state.opaque"]:
        return func, env
    new_func, new_env = run_pipeline(func, env, passes)
    envs = env['numba.state.envs']
    envs[new_func] = new_env
    return new_func, new_env

@cached('numba.typing', key=_cache_key)
def typing_phase(func, env, passes=typing):
    return run_pipeline(func, env, passes)

@cached('numba.opt')
def optimization_phase(func, env, passes=optimizations, dependences=None):
    envs = env["numba.state.envs"]
    if dependences is None:
        dependences = env["numba.state.dependences"]

    for f in dependences:
        if f != func:
            optimization_phase(f, envs[f], passes, [])
    run_pipeline(func, env, passes)

    return func, env

@cached('numba.prelowering')
def prelowering_phase(func, env, passes=prelowering, dependences=None):
    envs = env["numba.state.envs"]
    if dependences is None:
        dependences = env["numba.state.dependences"]

    for f in dependences:
        if f != func:
            prelowering_phase(f, envs[f], passes, [])
    run_pipeline(func, env, passes)

    return func, env

@cached('numba.lowering')
def lowering_phase(func, env, passes=lowering, dependences=None):
    envs = env["numba.state.envs"]
    if dependences is None:
        dependences = env["numba.state.dependences"]

    for f in dependences:
        if f != func:
            lowering_phase(f, envs[f], passes, [])
    run_pipeline(func, env, passes)

    return func, env

def codegen_phase(func, env):
    assert env['numba.target'] == 'cpu'
    cache = env['numba.codegen.cache']
    envs = env["numba.state.envs"]

    if func in cache:
        return cache[func]

    dependences = [d for d in env['numba.state.dependences']
                   if d not in cache]

    for f in dependences:
        run_pipeline(f, envs[f], backend_init)
    for f in dependences:
        run_pipeline(f, envs[f], backend_run)
    for f in dependences:
        e = envs[f]
        lfunc = e["numba.state.llvm_func"]
        run_pipeline(lfunc, envs[f], backend_finalize)

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
opt = phasecompose(optimization_phase, typing)
prelower = phasecompose(prelowering_phase, opt)
lower = phasecompose(lowering_phase, prelower)
cpu_codegen = phasecompose(codegen_phase, lower)

# ______________________________________________________________________
# Data Parallel Python Specifics

def dpp_codegen_phase(func, env):
    from pykit.codegen.llvm import llvm_utils
    assert env['numba.target'] == 'dpp'
    cache = env['numba.codegen.cache']
    envs = env["numba.state.envs"]

    if func in cache:
        return cache[func]

    dependences = [d for d in env['numba.state.dependences']
                   if d not in cache]

    for f in dependences:
        localenv = envs[f]
        localenv['codegen.llvm.module'] = llvm_utils.module("tmp.%x" % id(f))
        run_pipeline(f, envs[f], backend_init)

    for f in dependences:
        run_pipeline(f, envs[f], dpp_backend_run)

    for f in dependences:
        e = envs[f]
        lfunc = e["numba.state.llvm_func"]
        run_pipeline(lfunc, envs[f], dpp_backend_finalize)
        cache.insert(f, (lfunc, e))

    return env["numba.state.llvm_func"], env

dpp_lower = lower  # share cpu lowering for now
dpp_codegen = phasecompose(dpp_codegen_phase, dpp_lower)


# ______________________________________________________________________
# Naming

phases = {
    "setup":        setup,
    "translation":  translation,
    "typing":       typing,
    "opt":          opt,
    "lower":        lower,
    "codegen":      cpu_codegen,
}

# ______________________________________________________________________
# Apply

def apply_phase(phase, nb_func, argtypes):
    env = fresh_env(nb_func, argtypes)
    return phase(nb_func, env)

# ______________________________________________________________________
# Codegen

_target_codegen_map = {
    'cpu': cpu_codegen,
    'dpp': dpp_codegen,
}

def codegen(func, env):
    return _target_codegen_map[env['numba.target']](func, env)

