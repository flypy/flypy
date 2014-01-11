# -*- coding: utf-8 -*-

"""
Compiler utilities.
"""

from __future__ import print_function, division, absolute_import

from flypy.runtime.obj.core import Constructor

from pykit import types as ptypes
from pykit.ir import Function, OConst

class Caller(object):
    """
    Utility to call functions.
    """

    def __init__(self, builder, context, env):
        self.builder = builder
        self.context = context
        self.env = env

    def call(self, phase_to_run, nb_func, args, argtypes=None, result=None):
        from flypy.pipeline import phase

        # Apply phase
        if argtypes is None:
            argtypes = tuple(self.context[arg] for arg in args)

        target = self.env['flypy.target']
        f, env = phase.apply_phase(phase_to_run, nb_func, argtypes, target)

        # Generate call
        result = self.builder.call(ptypes.Opaque, f, args, result=result)

        # Update context
        self.context[result] = env["flypy.typing.restype"]

        return result

    def apply_constructor(self, ctor, args=()):
        """
        Apply a constructor with the given context, builder, constructor and
        arguments.
        """
        irctor = OConst(ctor)
        result = self.builder.call(ptypes.Opaque, irctor, list(args))

        argtypes = tuple(self.context[arg] for arg in args)
        self.context[ctor] = Constructor[ctor.type]
        self.context[result] = ctor[argtypes]

        return result



def callmap(f, func, env):
    """
    Map `f` over all calls in the function `func`.
    """
    context = env['flypy.typing.context']

    for op in func.ops:
        if op.opcode == 'call':
            f(context, op)

def jitcallmap(f, func, env):
    """
    Map `f` over all calls in the function `func` which are non-opaque
    jit functions.
    """
    context = env['flypy.typing.context']
    envs = env['flypy.state.envs']

    for op in func.ops:
        if op.opcode == 'call':
            callee, args = op.args
            if not isinstance(callee, Function):
                continue

            f_env = envs[callee]

            # Retrieve Python version and opaqueness
            py_func = f_env['flypy.state.py_func']
            opaque = f_env['flypy.state.opaque']

            if py_func and not opaque:
                f(context, py_func, f_env, op)

