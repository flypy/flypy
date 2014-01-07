# -*- coding: utf-8 -*-

"""
Constant Folding, used for sparse conditional constant propagation.
"""

from __future__ import print_function, division, absolute_import

from numba2.compiler.typing import infer_call
from pykit.optimizations import sccp

#===------------------------------------------------------------------===
# Constant Folding
#===------------------------------------------------------------------===

class ConstantFolder(sccp.ConstantFolder):
    """
    Fold calls to user-defined constants that are annotated with
    `eval_if_const`.
    """

    def __init__(self, context, env):
        self.context = context
        self.env = env
        self.envs = env['numba.state.envs']

    def op_call(self, op, cells):
        # Fetch the message receiver and the arguments
        func, args = op.args

        # Fetch all inputs
        consts = [cells[arg] for arg in args]

        if not all(map(sccp.isconst, consts)):
            return None # Some runtime values, bail

        # Determine constant result for all receivers
        result = None
        first = True
        method_types = self.context[func]
        for method_type in method_types:
            wrapper, obj = method_type.parameters
            signature = self.context[func]

            argtypes = [self.context[a] for a in args]
            typed_func, restype = infer_call(func, signature, argtypes)

            # Check whether we can safely evaluate the function
            func_env = self.envs[func]
            options = func_env['numba.state.options']
            if not options.get('eval_if_const'):
                return None

            # Evaluate
            const = func(*map(sccp.unwrap, consts))

            # Check result type
            ty = typeof(const)
            if ty not in self.context[op]:
                raise TypeError(
                    "Result %s of type %s does not agree with inferred "
                    "types %s" % (const, ty, self.context[op]))

            if first:
                first = False
                result = const
            elif const != result:
                return None

        return result