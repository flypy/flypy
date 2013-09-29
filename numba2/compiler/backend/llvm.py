# -*- coding: utf-8 -*-

"""
Numba compiler backend leveraging pykit.
"""

from __future__ import print_function, division, absolute_import

from pykit.codegen.llvm import llvm_codegen
from pykit.codegen import llvm
import llvm.core as lc

def codegen_init(func, env):
    """
    Initialize the code generator by allocating an LLVM function.
    """
    llvm_func = llvm_codegen.initialize(func, env)
    env["numba.state.llvm_func"] = llvm_func

    cache = env["codegen.cache"]
    cache[func] = llvm_func

    return func, env

def codegen_run(func, env):
    """
    Back-end entry point.

    Parameters
    ----------

    func : pykit.ir.Function
        Typed pykit function.

    Returns : llvm.core.Function
        Compiled and optimized llvm function.
    """
    lfunc = llvm_codegen.translate(func, env, env["numba.state.llvm_func"])
    return lfunc, env

#global_mod = lc.Module.new("global_module")

def codegen_link(func, env):
    """
    Link the llvm function into
    """
    lfunc = env["numba.state.llvm_func"]
    return lfunc, env

    #global_mod.link_in(lfunc.module, preserve=True)
    #new_lfunc = global_mod.get_function_named(lfunc.name)
    #
    #env["numba.state.llvm_func"] = new_lfunc
    #return new_lfunc, env

get_ctypes = llvm.get_ctypes