# -*- coding: utf-8 -*-

"""
Root finding on the stack. Here we are using simple Henderson shadow stacks [0].
Later, we probably want to use LLVM stack maps.

[0]: Accurate Garbage Collection in an Uncooperative Environment,
     Fergus Henderson, 2002
"""

from __future__ import print_function, division, absolute_import

from flypy import jit, sjit, cast, sizeof, NULL


@sjit('StackFrame[roots]')
class StackFrame(object):
    """
    Stack-allocated frame holding roots and per-type traverse functions.

    Note that we could generate a trace function for each function that
    hold heap pointers, but we consider that too expensive. We instead
    generate a table containing a trace function for each root.
    """

    layout = [
        ('prev', 'Pointer[StackFrame[]]'),
        ('trace_funcs', 'Pointer[Pointer[void]]'),
        ('roots', 'roots'), # Array[Pointer[int8]]
    ]

    @jit
    def __init__(self, prev):
        self.prev = prev


@jit('Pointer[StackFrame[]] -> r')
def find_roots(top_frame):
    """Find all roots in the stack along with their trace functions"""
    while top_frame != NULL:
        roots = top_frame[0].roots
        trace_funcs = top_frame[0].trace_funcs
        for i in range(len(roots)):
            if trace_funcs[i] == NULL:
                break
            if roots[i] != NULL:
                yield roots[i], trace_funcs[i]

        top_frame = top_frame[0].prev