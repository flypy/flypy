# -*- coding: utf-8 -*-

"""
Handle constructors.
"""

from __future__ import print_function, division, absolute_import

from numba2 import is_numba_type, int64
from numba2.environment import fresh_env
from numba2.representation import stack_allocate
from numba2.runtime.obj import Type
from numba2.runtime import gc

from pykit import types as ptypes
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

        call(C, x, y) -> call(C.__init__, x, y)
    """
    from numba2 import phase

    context = env['numba.typing.context']
    b = OpBuilder()

    for op in func.ops:
        if op.opcode == 'call':
            cls, args = op.args
            if isinstance(cls, Const) and is_numba_type(cls.const):
                cls = cls.const
                f = cls.__init__
                type = context[op]
                argtypes = [type] + [context[arg] for arg in op.args[1]]

                # TODO: implement this on Type.__call__ when we support *args
                e = fresh_env(f, argtypes)
                __init__, _ = phase.typing(f, e)

                alloc = allocate_object(b, type, env)
                call = b.call(ptypes.Void, __init__, [alloc] + args)

                op.replace_uses(alloc)
                op.replace([alloc, call])

                context[alloc] = type


def allocate_object(builder, type, env):
    """
    Allocate object of type `type`.
    """
    if stack_allocate(type):
        return builder.alloca(ptypes.Pointer(ptypes.Opaque))
    else:
        return heap_allocate(builder, type, env)


def heap_allocate(builder, type, env):
    """
    Heap allocate an object of type `type`
    """
    from numba2 import phase

    # TODO: implement untyped pykit builder !

    # Put object on the heap: call gc.gc_alloc(nitems, type)
    gcmod = gc.gc_impl(env["numba.gc.impl"])
    context = env['numba.typing.context']

    # Build arguments for gc_alloc
    n = Const(1, ptypes.Opaque)
    ty = Const(type, ptypes.Opaque)

    # Update type context with new constants
    context[n] = int64
    context[ty] = Type[type]

    # Type the gc_alloc function (TODO: (indirect) allocation of a
    # new object would recurse infinitely)
    f, _ = phase.apply_phase(phase.typing, gcmod.gc_alloc,
                             (int64, Type[type]))
    return builder.call(ptypes.Opaque, f, [n, ty])
