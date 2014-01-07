# -*- coding: utf-8 -*-

"""
Numba compilation environment.
"""

from __future__ import print_function, division, absolute_import

from numba2.utils import FrozenDict
from numba2.caching import Cache, InferenceCache, TypingCache

from pykit import environment as pykit_env
from pykit.codegen import llvm as llvm_codegen

#===------------------------------------------------------------------===
# CPU Environment
#===------------------------------------------------------------------===

_cpu_env = {
    # Command line args
    'numba.cmdopts':            {},
    'numba.script':             False, # True when run from the numba script

    # Caching
    'numba.initialize.cache':   Cache(),
    'numba.frontend.cache':     Cache(),
    'numba.typing.cache':       TypingCache(),
    'numba.inference.cache':    InferenceCache(),
    'numba.generators.cache':   Cache(),
    'numba.hl_lower.cache':     Cache(),
    'numba.opt.cache':          Cache(),
    'numba.prelower.cache':     Cache(),
    'numba.ll_lower.cache':     Cache(),
    'numba.dpp_codegen.cache':  Cache(),
    'numba.llvm.cache':         Cache(),
    'numba.codegen.cache':      Cache(),

    # General state
    'numba.state.func_name':    None,   # Function name
    'numba.state.modname':      None,   # Module name
    'numba.state.qname':        None,   # Qualified name inside module
    'numba.state.py_func':      None,   # This value may be None
    'numba.state.func_globals': None,
    'numba.state.func_code':    None,
    'numba.state.callgraph':    None,   # TODO: unused ...
    'numba.state.opaque':       False,  # Whether the function is opaque
    'numba.state.generator':    0,      # Counts the number of 'yield' exprs
    'numba.state.phase':        None,
    'numba.state.copies':       None,
    'numba.state.crnt_func':    None,
    'numba.state.options':      None,
    'numba.state.dependence':   None,

    # GC
    'numba.gc.impl':            "boehm",

    # Global state
    'numba.state.envs':         {},     # All cached environments

    # Typing
    'numba.typing.restype': None,       # Input/Output
    'numba.typing.argtypes': None,      # Input
    'numba.typing.signature': None,     # Output
    'numba.typing.context': None,       # Output
    'numba.typing.constraints': None,   # Output

    # Flags
    'numba.verify':         True,
    'numba.optimize':       True,
    'numba.target':         'cpu',

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
    'numba.typing.cache':       TypingCache(),
    'numba.inference.cache':    InferenceCache(),
    'numba.opt.cache':          Cache(),
    'numba.prelowering.cache':  Cache(),
    'numba.lowering.cache':     Cache(),
    'numba.codegen.cache':      Cache(),

    "numba.target": "dpp",

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

def fresh_env(func, argtypes, target="cpu"):
    """
    Allocate a new environment.
    """
    env = dict(_target_env_map[target])
    py_func = func.py_func

    # Types
    env['numba.typing.argtypes'] = argtypes

    # State
    env["numba.state.function_wrapper"] = func
    env['numba.state.py_func'] = py_func
    env['numba.state.func_name'] = py_func.__module__
    env['numba.state.func_module'] = py_func.__module__
    env['numba.state.func_qname'] = py_func.__name__
    env['numba.state.func_globals'] = py_func.__globals__
    env['numba.state.func_code'] = py_func.__code__

    # Copy
    def copy(func1, argtypes1):
        return fresh_env(func1, argtypes1, target)

    env['numba.fresh_env'] = copy

    return env
