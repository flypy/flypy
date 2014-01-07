# -*- coding: utf-8 -*-

"""
flypy optimizations.
"""

from __future__ import print_function, division, absolute_import

from pykit import pipeline

def optimize(func, env):
    cache = env['flypy.opt.cache']
    optimize = env['flypy.optimize']

    optimized = cache.lookup((func, optimize))
    if not optimized:
        #analyzed, env = pipeline.analyze(func, env)
        optimized, env = pipeline.optimize(func, env)

    return optimized, env

run = optimize