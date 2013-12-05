# -*- coding: utf-8 -*-

"""
Check variable use and scoping rules.
"""

from __future__ import print_function, division, absolute_import

from numba2.errors import error_context_phase

from pykit.ir import Undef
from pykit.utils import flatten

#===------------------------------------------------------------------===
# Scoping
#===------------------------------------------------------------------===

def check_scoping(func, env):
    for op in func.ops:
        if op.opcode == 'phi':
            continue

        for arg in flatten(op.args):
            if isinstance(arg, Undef):
                raise NameError("Variable referenced before assignment")


def run(func, env):
    from numba2.pipeline import phase

    with error_context_phase(env, phase.typing):
        check_scoping(func, env)