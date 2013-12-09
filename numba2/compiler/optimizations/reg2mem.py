# -*- coding: utf-8 -*-

"""
Translate out of SSA.
"""

from __future__ import print_function, division, absolute_import
from pykit.transform import reg2mem
from .utils import update_context

# TODO: This is not an optimization, move it elsewhere?

def run(func, env):
    #if not env.get('numba.generator.consumers'):
    #    return
    vars, loads = reg2mem.reg2mem(func, env)
    context = env['numba.typing.context']

    if context is not None:
        update_context(env, env, vars)

        for phi in loads:
            for newop in loads[phi]:
                context[newop] = context[phi]