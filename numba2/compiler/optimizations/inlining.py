# -*- coding: utf-8 -*-

"""
Inlining of numba functions with 'inline=True'. See also the `ijit`
decorator.
"""

from __future__ import print_function, division, absolute_import

from pykit.ir import Function, Op
from pykit.transform import inline

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
                    valuemap = inline.inline(func, op)
                    update_context(env, e, valuemap)
                    changed = True

    return changed

def update_context(env, callee_env, valuemap):
    """
    Update our typing context with the types of the new inlined operations.
    """
    context = env['numba.typing.context']
    callee_context = callee_env['numba.typing.context']

    for old_op, new_op in valuemap.iteritems():
        if old_op in callee_context:
            context[new_op] = callee_context[old_op]

    for const, type in callee_context.iteritems():
        if not isinstance(const, Op):
            context[const] = type