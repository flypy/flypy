# -*- coding: utf-8 -*-

"""
Object allocation. Lower to GC or stack-allocation based on available
information.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from flypy import is_flypy_type, int64, errors
from flypy.compiler.utils import Caller
from flypy.types import Type, Pointer, void
from flypy.representation import stack_allocate
from flypy.runtime import gc

from pykit import types as ptypes
from pykit import ir
from pykit.ir import Builder, OpBuilder, Const

def allocator(func, env):
    context = env['flypy.typing.context']
    b = OpBuilder()
    caller = Caller(b, context, env)
    gcmod = gc.gc_impl(env["flypy.gc.impl"])

    for op in func.ops:
        newop = None
        if op.opcode == 'allocate_obj':
            stmts, newop = allocate_object(caller, b, context[op], env)
            newop.result = op.result
        elif op.opcode == 'register_finalizer':
            stmts = register_finalizer(caller, b, env, context,
                                       context[op.args[0]], gcmod, op.args[0])
        else:
            continue

        if stmts:
            op.replace(stmts)
        else:
            op.delete()


def allocate_object(caller, builder, type, env):
    """
    Allocate object of type `type`.
    """
    if stack_allocate(type):
        obj = builder.alloca(ptypes.Pointer(ptypes.Opaque))
        return [obj], obj
    else:
        if env['flypy.target'] != 'cpu':
            raise errors.CompileError(
                "Cannot heap allocate object of type %s with target %r" % (
                                                type, env['flypy.target']))
        return heap_allocate(caller, builder, type, env)

# TODO: generating calls or typed codes is still messy:
# TODO:     - write "untyped" pykit builder
# TODO:     - write typed flypy builder (?)

def heap_allocate(caller, builder, type, env):
    """
    Heap allocate an object of type `type`
    """
    phase = env['flypy.state.phase']
    # TODO: implement untyped pykit builder !

    # Put object on the heap: call gc.gc_alloc(nitems, type)
    gcmod = gc.gc_impl(env["flypy.gc.impl"])
    context = env['flypy.typing.context']

    # Build arguments for gc_alloc
    n = Const(1, ptypes.Opaque)
    ty = Const(type, ptypes.Opaque)
    context[n] = int64
    context[ty] = Type[type]

    # Type the gc_alloc function
    p = caller.call(phase, gcmod.gc_alloc, [n, ty])
    obj = builder.convert(ptypes.Opaque, p)

    # Update type context
    context[p] = Pointer[void]

    return [p, obj], obj

def register_finalizer(caller, builder, env, context, type, gcmod, obj):
    """
    Register a finalizer for the object given as pointer `obj`.
    """
    from flypy.pipeline import phase
    curphase = env['flypy.state.phase']

    #(TODO: (indirect) allocation of a new object in __del__ will recurse
    # infinitely)

    if '__del__' in type.fields:
        # Compile __del__
        __del__ = type.fields['__del__']
        lfunc, env = phase.apply_phase(phase.codegen, __del__, (type,),
                                       env['flypy.target'])

        # Retrieve function address of __del__
        cfunc = env["codegen.llvm.ctypes"]
        pointerval = ctypes.cast(cfunc, ctypes.c_void_p).value
        ptr = ir.Pointer(pointerval, ptypes.Pointer(ptypes.Void))
        context[ptr] = Pointer[void]

        # Cast object to void *
        obj_p = builder.convert(ptypes.Opaque, obj)
        context[obj_p] = Pointer[void]

        # Call gc_add_finalizer with (obj, ptr)
        result = caller.call(curphase, gcmod.gc_add_finalizer, [obj_p, ptr])
        context[result] = void

        return [obj_p, result]
