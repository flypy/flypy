# -*- coding: utf-8 -*-

"""
Rewrite exceptions that are thrown and caught locally to jumps.
"""

from flypy.compiler import excmodel

from pykit.analysis import cfa
from pykit.optimizations import local_exceptions

def rewrite_local_exceptions(func, env):
    """
    Rewrite exc_throw(exc) -> jump(handler_block) for statically determined
    exceptions.
    """
    local_exceptions.run(func, env, exc_model=excmodel.ExcModel(env))


def rewrite_exceptions(func, env):
    blocks = set()
    for op in func.ops:
        if op.opcode == 'exc_throw':
            raise NotImplementedError("Exception throwing", op, func)
        if op.opcode in ('exc_catch', 'exc_setup'):
            blocks.add(op.block)
            op.delete()

    update_outdated_incoming_blocks(func, blocks)

def update_outdated_incoming_blocks(func, candidates):
    """
    Update phi nodes in blocks previously containing 'exc_catch'. 'exc_setup'
    may span many blocks, and none, or only a subset of those blocks may be
    actual predecessors.
    """
    cfg = cfa.cfg(func)
    for block in candidates:
        preds = cfg.predecessors(block)
        for op in block.leaders:
            if op.opcode == 'phi':
                blocks, values = op.args
                newblocks = [block for block in blocks if block in preds]
                newvalues = [val for block, val in zip(blocks, values)
                                     if block in preds]
                assert len(newblocks) == len(preds), (op.block, newblocks,
                                                      preds, blocks)
                op.set_args([newblocks, newvalues])