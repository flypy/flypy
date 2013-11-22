# -*- coding: utf-8 -*-

"""
Pretty printing of numba IRs.
"""

from __future__ import print_function, division, absolute_import

import os
import re
import dis
import types
from functools import wraps, partial

from .lexing import lex_source
from . import viz
from numba2 import pipeline

import pykit.ir
from pykit.analysis import cfa

#===------------------------------------------------------------------===
# Passes
#===------------------------------------------------------------------===

def dumppass(option):
    """Apply `option` if it is active in the cmdops of the environment."""
    def decorator(f):
        @wraps(f)
        def wrapper(func, env):
            cmdopts = env['numba.cmdopts']
            if cmdopts.get(option):
                return f(func, env, cmdopts.get("fancy"))
        return wrapper
    return decorator

# ______________________________________________________________________

#@dumppass("dump")
def dump(func, env):
    print(func)

@dumppass("dump-cfg")
def dump_cfg(func, env, fancy):
    CFG = cfa.cfg(func)
    viz.dump(CFG.nx, os.path.expanduser("~/cfg.dot"))

@dumppass("dump-llvm")
def dump_llvm(func, env, fancy):
    print(lex_source(str(func), "llvm", "console"))

@dumppass("dump-optimized")
def dump_optimized(func, env, fancy):
    print(lex_source(str(func), "llvm", "console"))

#===------------------------------------------------------------------===
# Verbose
#===------------------------------------------------------------------===

def augment_pipeline(passes):
    return [partial(verbose, p) for p in passes]

def debug_print(func, env):
    """
    Returns whether to enable debug printing, checks the '--filter' argument
    to './bin/numba'
    """
    cmdopts = env['numba.cmdopts']
    if cmdopts and cmdopts['filter']:
        func_name = env["numba.state.func_name"]
        return re.search(cmdopts['filter'], func_name)
    return env['numba.script']

def verbose(p, func, env):
    if not debug_print(func, env):
        return pipeline.apply_transform(p, func, env)

    argtypes = env['numba.typing.argtypes']
    title = "%s [ %s(%s) ]" % (_passname(p), _funcname(func),
                               ", ".join(map(str, argtypes)))

    print(title.center(60).center(90, "-"))

    if isinstance(func, types.FunctionType):
        dis.dis(func)
        func, env = pipeline.apply_transform(p, func, env)
        print()
        print(func)
        return func, env

    before = _formatfunc(func)
    func, env = pipeline.apply_transform(p, func, env)
    after = _formatfunc(func)

    if before != after:
        print(pykit.ir.diff(before, after))

    return func, env

# ______________________________________________________________________

def _passname(transform):
    return transform.__name__
    #if isinstance(transform, types.ModuleType):
    #    return transform.__name__
    #else:
    #    return ".".join([transform.__module__, transform.__name__])

def _funcname(func):
    if isinstance(func, types.FunctionType):
        return func.__name__
    else:
        return func.name

def _formatfunc(func):
    if isinstance(func, types.FunctionType):
        dis.dis(func)
        return ""
    else:
        return str(func)