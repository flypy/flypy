# -*- coding: utf-8 -*-

"""
Rewrite exceptions that are thrown and caught locally to jumps.
"""

from numba2.compiler import excmodel
from pykit.optimizations import local_exceptions

def run(func, env):
    local_exceptions.run(func, env, exc_model=excmodel.ExcModel(env))
