# -*- coding: utf-8 -*-

"""
Numba compiler phases (groupings of passes).
"""

from __future__ import print_function, division, absolute_import

import os
from functools import partial, wraps

from numba2.viz.viz import dump
from .pipeline import run_pipeline
from . import passes
from .environment import fresh_env

from pykit.analysis import callgraph

#===------------------------------------------------------------------===
# Phases
#===------------------------------------------------------------------===

phases = {
    # phase name -> phase
}

phase_passes = {
    # phase -> passes
}

def define_phase(phase_func, phase_name, passes):
    phases[phase_name] = phase_func
    phase_passes[phase_func] = passes
    return phase_func

class Phase(object):

    def __init__(self, phase_name, passes, all, skip_opaque, cache_key):
        self.phase_name = phase_name
        self.passes = passes
        self.all = all
        self.skip_opaque = skip_opaque
        self.cache_key = cache_key

    def apply(self, func, env):
        if self.all:
            return apply_all(self, func, env)
        else:
            return self.apply_single(func, env)

    def apply_single(self, func, env):
        # -- Lookup cached version -- #
        result = lookup_cached(self.phase_name, self.cache_key, func, env)
        if result is not None:
            return result
        elif self.skip_opaque and env["numba.state.opaque"]:
            return func, env

        # -- Apply phase -- #
        env['numba.state.phase'] = phases[self.phase_name]
        new_func, new_env = run_pipeline(func, env, self.passes)

        # -- Update cache -- #
        insert_cache(self.phase_name, self.cache_key,
                     func, env, new_func, new_env)
        return new_func, new_env

    def __repr__(self):
        return "phase(%s)" % (self.phase_name,)

    @property
    def __name__(self):
        return self.phase_name

    __call__ = apply


def phase(phase_name, passes, depend=None, skip_opaque=False, all=True, key=None):
    """
    Define a phase.
    """
    phase = Phase(phase_name, passes, all, skip_opaque, key or cache_key)
    phase_passes[phase] = passes
    if depend:
        phase = phasecompose(phase, depend)
    return define_phase(phase, phase_name, passes)

# ______________________________________________________________________
# Phase application

def apply_all(phase, func, env, dependences=None):
    """Apply a phase to a function and its dependences"""
    if dependences is None:
        dependences = callgraph.callgraph(func).node

    envs = env["numba.state.envs"]
    for dep in dependences:
        if func != dep:
            dep_env = envs[dep]
            phase.apply_single(dep, dep_env)

    return phase.apply_single(func, env)

def apply_phase(phase, nb_func, argtypes, target):
    env = fresh_env(nb_func, argtypes, target)
    return phase(nb_func, env)

# ______________________________________________________________________
# Caching

def get_cache(phase_name, env):
    cache_name = '.'.join(['numba', phase_name, 'cache'])
    return env[cache_name]

def lookup_cached(phase_name, key, func, env):
    cache = get_cache(phase_name, env)
    cache_key = key(func, env)

    if cache.lookup(cache_key):
        new_func, new_env = cache.lookup(cache_key)
        if phase_name in ('frontend',):
            # TODO: This is a hack! Do manual caching in
            # translation_phase below!
            new_env = env # Don't lose the argtypes
        return new_func, new_env

def insert_cache(phase_name, cache_key, func, env, new_func, new_env):
    cache = get_cache(phase_name, env)
    key = cache_key(func, env)
    cache.insert(key, (new_func, new_env))

def phasecompose(phase1, phase2):
    result = lambda func, env: phase1(*phase2(func, env))
    return wraps(phase1)(result)

def cache_key(func, env):
    return func

def cache_key_argtypes(func, env):
    return (func, tuple(env["numba.typing.argtypes"]))

# ______________________________________________________________________
# Utils

def _deps(func, debug=False):
    """
    Locate dependences for a function.
    """
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
# phases

def llvm_phase(func, env):
    cache = env['numba.codegen.cache']
    envs = env["numba.state.envs"]

    if func in cache:
        return cache[func]

    dependences = [d for d in _deps(func, debug=False) if d not in cache]

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

# Data Parallel Python Specifics

def dpp_llvm_phase(func, env):
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
        run_pipeline(f, envs[f], passes.backend_init)

    for f in dependences:
        run_pipeline(f, envs[f], passes.dpp_backend_run)

    for f in dependences:
        e = envs[f]
        lfunc = e["numba.state.llvm_func"]
        run_pipeline(lfunc, envs[f], passes.dpp_backend_finalize)
        cache.insert(f, (lfunc, e))

    return env["numba.state.llvm_func"], env

# ______________________________________________________________________
# Apply

initialize  = phase('initialize', passes.initialize, all=False,
                    key=cache_key_argtypes)
frontend    = phase('frontend', passes.frontend, depend=initialize,
                    skip_opaque=True, all=False, key=cache_key_argtypes)
typing      = phase('typing', passes.typing, depend=frontend, all=False,
                    key=cache_key_argtypes)
generators  = phase('generators', passes.generators, depend=typing, all=False)
hl_lower    = phase('hl_lower', passes.hl_lowering, depend=generators)
opt         = phase('opt', passes.optimizations, depend=hl_lower)
prelower    = phase('prelower', passes.prelowering, depend=opt)
ll_lower    = phase('ll_lower', passes.ll_lowering, depend=prelower)
llvm        = phasecompose(llvm_phase, ll_lower)
cpu_codegen = phase('codegen', passes.codegen, depend=llvm, all=False)

dpp_llvm    = phasecompose(dpp_llvm_phase, ll_lower)
dpp_codegen = phase('dpp_codegen', passes.codegen, depend=dpp_llvm, all=False)

# ______________________________________________________________________
# Codegen

_target_codegen_map = {
    'cpu': cpu_codegen,
    'dpp': dpp_codegen,
}

def codegen(func, env):
    return _target_codegen_map[env['numba.target']](func, env)
