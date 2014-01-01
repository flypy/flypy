# -*- coding: utf-8 -*-

"""
Compiler utilities.
"""

from __future__ import print_function, division, absolute_import

from pykit import types as ptypes

class Caller(object):
    """
    Utility to call functions.
    """

    def __init__(self, builder, context, env):
        self.builder = builder
        self.context = context
        self.env = env

    def call(self, phase_to_run, nb_func, args, argtypes=None, result=None):
        from numba2.pipeline import phase

        # Apply phase
        if argtypes is None:
            argtypes = tuple(self.context[arg] for arg in args)

        target = self.env['numba.target']
        f, env = phase.apply_phase(phase_to_run, nb_func, argtypes, target)

        # Generate call
        result = self.builder.call(ptypes.Opaque, f, args, result=result)

        # Update context
        self.context[result] = env["numba.typing.restype"]

        return result