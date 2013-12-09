# -*- coding: utf-8 -*-

"""
Optimization utilities.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import Op


def update_context(dst_env, src_env, valuemap):
    """
    Update our typing context with the types from the original operations
    and context.
    """
    context = dst_env['numba.typing.context']
    callee_context = src_env['numba.typing.context']

    for old_op, new_op in valuemap.iteritems():
        if old_op in callee_context:
            context[new_op] = callee_context[old_op]

    for const, type in callee_context.iteritems():
        if not isinstance(const, Op):
            context[const] = type