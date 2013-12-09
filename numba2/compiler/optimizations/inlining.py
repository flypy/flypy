# -*- coding: utf-8 -*-

"""
Inlining of numba functions with 'inline=True'. See also the `ijit`
decorator.
"""

from __future__ import print_function, division, absolute_import

from pykit.ir import Function, Op
from pykit.transform import inline

from .utils import update_context

#===------------------------------------------------------------------===
# Pass
#===------------------------------------------------------------------===

def run(func, env):
    """
    Inline numba functions with 'inline=True'
    """
    changed = True
    while changed:
        changed = inliner(func, env)

def inliner(func, env):
    envs = env['numba.state.envs']
    changed = False

    for op in func.ops:
        if op.opcode == 'call':
            # See if we are messaging a static receiver
            f, args = op.args
            if isinstance(f, Function):
                # See if `f` was defined with `inline=True`
                e = envs[f]
                options = e['numba.state.options']
                if options.get('inline'):
                    inline_callee(func, op, env, e)
                    changed = True

    return changed


def inline_callee(func, call, env, callee_env):
    valuemap = inline.inline(func, call)
    update_context(env, callee_env, valuemap)
    return valuemap
