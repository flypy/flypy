# -*- coding: utf-8 -*-

"""
Inlining of numba functions with 'inline=True'. See also the `ijit`
decorator.
"""

from __future__ import print_function, division, absolute_import

from pykit.ir import Function
from pykit.transform import inline

#===------------------------------------------------------------------===
# Pass
#===------------------------------------------------------------------===

def inliner(func, env):
    """
    Inline numba functions with 'inline=True'
    """
    envs = env['numba.state.envs']

    for op in func.ops:
        if op.opcode == 'call':
            # See if we are messaging a static receiver
            f, args = op.args
            if isinstance(f, Function):
                # See if `f` was defined with `inline=True`
                e = envs[f]
                options = e['numba.state.options']
                if options.get('inline'):
                    inline.inline(func, op)

run = inliner