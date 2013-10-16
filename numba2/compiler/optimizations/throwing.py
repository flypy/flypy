# -*- coding: utf-8 -*-

"""
Rewrite exceptions that are thrown and caught locally to jumps.
"""

from numba2.compiler import excmodel
from pykit.optimizations import local_exceptions

def rewrite_local_exceptions(func, env):
    local_exceptions.run(func, env, exc_model=excmodel.ExcModel(env))

def rewrite_exceptions(func, env):
    for op in func.ops:
        if op.opcode == 'exc_throw':
            raise NotImplementedError("Exception throwing", op, func)
        if op.opcode in ('exc_catch', 'exc_setup'):
            op.delete()
