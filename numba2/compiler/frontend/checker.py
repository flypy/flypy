# -*- coding: utf-8 -*-

"""
Frontend IR checker.
"""

from __future__ import print_function, division, absolute_import

from numba2.errors import errctx, CompileError

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


def check_generators(func, env):
    for op in func.ops:
        if op.opcode == 'yield' and func.uses[op]:
            raise CompileError("yield expressions are not supported")


def run(func, env):
    from numba2.pipeline import phase

    with errctx(env):
        check_scoping(func, env)
        check_generators(func, env)