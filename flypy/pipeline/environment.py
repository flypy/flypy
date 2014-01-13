# -*- coding: utf-8 -*-

"""
flypy compilation environment.
"""

from __future__ import print_function, division, absolute_import

from flypy.utils import FrozenDict
from flypy.caching import Cache, InferenceCache, TypingCache

from pykit import environment as pykit_env
from pykit.codegen import llvm as llvm_codegen

#===------------------------------------------------------------------===
# CPU Environment
#===------------------------------------------------------------------===

_cpu_env = {
    # Command line args
    'flypy.cmdopts':            {},
    'flypy.script':             False, # True when run from the flypy script

    # Caching
    'flypy.initialize.cache':   Cache(),
    'flypy.frontend.cache':     Cache(),
    'flypy.typing.cache':       TypingCache(),
    'flypy.inference.cache':    InferenceCache(),
    'flypy.generators.cache':   Cache(),
    'flypy.hl_lower.cache':     Cache(),
    'flypy.opt.cache':          Cache(),
    'flypy.prelower.cache':     Cache(),
    'flypy.ll_lower.cache':     Cache(),
    'flypy.dpp_codegen.cache':  Cache(),
    'flypy.llvm.cache':         Cache(),
    'flypy.codegen.cache':      Cache(),

    # General state
    'flypy.state.func_name':    None,   # Function name
    'flypy.state.modname':      None,   # Module name
    'flypy.state.qname':        None,   # Qualified name inside module
    'flypy.state.py_func':      None,   # This value may be None
    'flypy.state.func_globals': None,
    'flypy.state.func_code':    None,
    'flypy.state.callgraph':    None,   # TODO: unused ...
    'flypy.state.opaque':       False,  # Whether the function is opaque
    'flypy.state.generator':    0,      # Counts the number of 'yield' exprs
    'flypy.state.phase':        None,
    'flypy.state.copies':       None,
    'flypy.state.crnt_func':    None,
    'flypy.state.options':      None,
    'flypy.state.call_flags':   None,   # Flags on how arguments are passed for each call Op
    'flypy.state.called_flags': None,   # How this function itself was called
    'flypy.state.dependence':   None,

    # GC
    'flypy.gc.impl':            "boehm",

    # Global state
    'flypy.state.envs':         {},     # All cached environments

    # Typing
    'flypy.typing.restype': None,       # Input/Output
    'flypy.typing.argtypes': None,      # Input
    'flypy.typing.signature': None,     # Output
    'flypy.typing.context': None,       # Output
    'flypy.typing.constraints': None,   # Output

    # Flags
    'flypy.verify':         True,
    'flypy.optimize':       True,
    'flypy.target':         'cpu',

    # Codegen
    "codegen.llvm.opt":     None,
    "codegen.llvm.engine":  None,
    "codegen.llvm.module":  None,
    "codegen.llvm.machine": None,
    "codegen.llvm.ctypes":  None,
}

_cpu_env.update(pykit_env.fresh_env())
llvm_codegen.install(_cpu_env)

cpu_env = FrozenDict(_cpu_env)

#===------------------------------------------------------------------===
# Data Parallel Python Environment
#===------------------------------------------------------------------===

_dpp_env = dict(cpu_env)
_dpp_env .update({
    'flypy.typing.cache':       TypingCache(),
    'flypy.inference.cache':    InferenceCache(),
    'flypy.opt.cache':          Cache(),
    'flypy.prelowering.cache':  Cache(),
    'flypy.lowering.cache':     Cache(),
    'flypy.codegen.cache':      Cache(),

    "flypy.target": "dpp",

    "codegen.llvm.opt":     None,
    "codegen.llvm.engine":  None,
    "codegen.llvm.module":  None,
    "codegen.llvm.machine": None,
    "codegen.llvm.ctypes":  None,
})
dpp_env = FrozenDict(_dpp_env)


_target_env_map = {
    'cpu': cpu_env,
    'dpp': dpp_env,
}

#===------------------------------------------------------------------===
# New envs
#===------------------------------------------------------------------===

def fresh_env(func, argtypes, target="cpu", varargs=False, keywords=False):
    """
    Allocate a new environment.
    """
    env = dict(_target_env_map[target])
    py_func = func.py_func

    # Types
    env['flypy.typing.argtypes'] = argtypes

    # State
    env["flypy.state.function_wrapper"] = func
    env['flypy.state.py_func'] = py_func
    env['flypy.state.func_name'] = py_func.__module__
    env['flypy.state.func_module'] = py_func.__module__
    env['flypy.state.func_qname'] = py_func.__name__
    env['flypy.state.func_globals'] = py_func.__globals__
    env['flypy.state.func_code'] = py_func.__code__
    env['flypy.state.called_flags'] = {'varargs': varargs, 'keywords': keywords}

    # Copy
    def copy(func1, argtypes1, **kwds):
        return fresh_env(func1, argtypes1, target, **kwds)

    env['flypy.fresh_env'] = copy

    return env
