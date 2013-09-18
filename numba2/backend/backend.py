# -*- coding: utf-8 -*-

"""
Numba compiler backend leveraging pykit.
"""

from __future__ import print_function, division, absolute_import

from pykit import pipeline, environment
from pykit.codegen import llvm

def run(func, env, codegen=llvm):
    """
    Back-end entry point.

    Parameters
    ----------

    func : pykit.ir.Function
        Typed pykit function.

    Returns : llvm.core.Function
        Compiled and optimized llvm function.
    """

    # -------------------------------------------------
    # Set up pykit environment

    env.update(environment.fresh_env())
    codegen.install(env)

    # -------------------------------------------------
    # Fire off pykit

    func, env = optimize(func, env)
    func, env = pipeline.codegen(func, env)

    return func, env

def optimize(func, env):
    cache = env['numba.opt.cache']
    optimize = env['numba.optimize']

    optimized = cache.lookup((func, optimize))
    if not optimized:
        analyzed, env = pipeline.analyze(func, env)
        optimized, env = pipeline.optimize(func, env)

    return optimized, env