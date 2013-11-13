# -*- coding: utf-8 -*-

"""
Handle constructors.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from numba2 import is_numba_type, int64, errors
from numba2.compiler.utils import Caller
from numba2.types import Type, Pointer, void
from numba2.environment import fresh_env
from numba2.representation import stack_allocate
from numba2.runtime import gc

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

        call(C, x, y) -> call(C.__init__, x, y)
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
                stmts, obj = allocate_object(caller, b, type, env)
                context[obj] = type

                # Initialize object (__init__)
                # TODO: implement this on Type.__call__ when we support *args
                call = caller.call(phase.typing, f, [obj] + op.args[1])

                op.replace_uses(obj)
                op.replace(stmts + [call])


def allocate_object(caller, builder, type, env):
    """
    Allocate object of type `type`.
    """
    if stack_allocate(type):
        obj = builder.alloca(ptypes.Pointer(ptypes.Opaque))
        return [obj], obj
    else:
        if env['numba.target'] != 'cpu':
            raise errors.CompileError(
                "Cannot heap allocate object of type %s with target %r" % (
                                                type, env['numba.target']))
        stmts, obj = heap_allocate(caller, builder, type, env)
        return stmts, obj

# TODO: generating calls or typed codes is still messy:
# TODO:     - write "untyped" pykit builder
# TODO:     - write typed numba builder (?)

def heap_allocate(caller, builder, type, env):
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
    context[n] = int64
    context[ty] = Type[type]

    # Type the gc_alloc function
    p = caller.call(phase.typing, gcmod.gc_alloc, [n, ty])
    obj = builder.convert(ptypes.Opaque, p)

    registered = register_finalizer(caller, builder, context, type, gcmod, p)

    # Update type context
    context[p] = Pointer[void]

    stmts = filter(None, [p, obj, registered])
    return stmts, obj

def register_finalizer(caller, builder, context, type, gcmod, obj):
    """
    Register a finalizer for the object given as pointer `obj`.
    """
    from numba2 import phase

    #(TODO: (indirect) allocation of a new object in __del__ will recurse
    # infinitely)

    if '__del__' in type.fields:
        # Compile __del__
        __del__ = type.fields['__del__']
        lfunc, env = phase.apply_phase(phase.codegen, __del__, (type,))

        # Retrieve function address of __del__
        cfunc = env["codegen.llvm.ctypes"]
        pointerval = ctypes.cast(cfunc, ctypes.c_void_p).value
        ptr = ir.Pointer(pointerval, ptypes.Pointer(ptypes.Void))
        context[ptr] = Pointer[void]

        # Call gc_add_finalizer with (obj, ptr)
        result = caller.call(phase.typing, gcmod.gc_add_finalizer, [obj, ptr])
        context[result] = void

        return result
