# -*- coding: utf-8 -*-

"""
Handle constructors.
"""

from __future__ import print_function, division, absolute_import

from numba2 import is_numba_type
from numba2.compiler.utils import Caller
from numba2.types import Type, void

from pykit import types as ptypes
from pykit import ir
from pykit.ir import Builder, OpBuilder, Const

def rewrite_raise_exc_type(func, env):
    """
    Rewrite 'raise Exception' to 'raise Exception()'
    """
    context = env['numba.typing.context']
    b = Builder(func)

    for op in func.ops:
        if op.opcode == 'exc_throw':
            [exc_type] = op.args
            if isinstance(exc_type, Const):
                ty = context[exc_type]
                if ty.impl == Type: # Type[Exception[]]
                    # Generate constructor application
                    b.position_before(op)
                    exc_obj = b.call(ptypes.Opaque, exc_type, [])
                    op.set_args([exc_obj])

                    type = ty.parameters[0]
                    context[exc_obj] = type


def rewrite_constructors(func, env):
    """
    Rewrite constructor application to object allocation followed by
    cls.__init__:

    Rewrite C(x, y) to:

        obj = allocate()
        C.__init__(obj, x, y)
        register_finalizer(obj.__del__)

    """
    from numba2 import phase

    context = env['numba.typing.context']
    b = OpBuilder()
    caller = Caller(b, context)

    for op in func.ops:
        if op.opcode == 'call':
            cls, args = op.args
            if isinstance(cls, Const) and is_numba_type(cls.const):
                cls = cls.const
                f = cls.__init__
                type = context[op]

                # Allocate object
                obj = ir.Op(
                    'allocate_obj', ptypes.Pointer(ptypes.Void), args=[])
                register_finalizer = ir.Op(
                    'register_finalizer', ptypes.Void, args=[obj])
                context[register_finalizer] = void
                context[obj] = type

                # Initialize object (call __init__)
                # TODO: implement this on Type.__call__ when we support *args
                initialize = caller.call(phase.typing, f, [obj] + op.args[1])

                op.replace_uses(obj)
                op.replace([obj, initialize, register_finalizer])
