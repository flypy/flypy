# -*- coding: utf-8 -*-

"""
Post-passes on untyped IR emitted by the front-end bytecode translation.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import Builder

#===------------------------------------------------------------------===
# Exceptions
#===------------------------------------------------------------------===

def simplify_exceptions(func, env=None):
    """
    Rewrite exceptions emitted by the front-end:

        exc_end -> split block
    """
    b = Builder(func)
    for op in func.ops:
        if op.opcode == 'exc_end':
            b.position_after(op)
            b.splitblock(terminate=True)
            op.delete()