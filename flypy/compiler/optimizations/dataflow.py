# -*- coding: utf-8 -*-

"""
Dataflow.
"""

from __future__ import print_function, division, absolute_import
from pykit.analysis import cfa
from pykit.ir import Undef

def dataflow(func, env, sync_context=True):
    """
    Move all allocas to the start block and then use pykit's SSA pass.
    We can move all allocas since all our objects are immutable
    """
    allocas = [op for op in func.ops if op.opcode == 'alloca']
    cfa.move_allocas(func, allocas)

    CFG = cfa.cfg(func)
    phis = cfa.ssa(func, CFG)

    if sync_context:
        context = env['flypy.typing.context']
        for phi, alloc in phis.items():
            type = context[alloc]
            context[phi] = type
            for arg in phi.args[1]:
                if isinstance(arg, Undef):
                    context[arg] = type

def run(func, env):
    dataflow(func, env, sync_context=False)