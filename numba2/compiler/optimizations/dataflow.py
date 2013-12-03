# -*- coding: utf-8 -*-

"""
Dataflow.
"""

from __future__ import print_function, division, absolute_import

from pykit.analysis import cfa

def run(func, env):
    """
    Move all allocas to the start block and then use pykit's SSA pass.
    We can move all allocas since all our objects are immutable
    """
    allocas = [op for op in func.ops if op.opcode == 'alloca']
    cfa.move_allocas(func, allocas)
    cfa.run(func, env)