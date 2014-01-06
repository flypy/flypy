# -*- coding: utf-8 -*-

"""
Preparation for codegen.
"""

from __future__ import print_function, division, absolute_import
from functools import partial

from numba2 import types, errors, conversion, compiler

from pykit.ir import FuncArg, Op, Const, Pointer, Struct, Undef
from pykit import types as ptypes
from pykit.utils import nestedmap

#===------------------------------------------------------------------===
# Types
#===------------------------------------------------------------------===

dummy_type = [('dummy',), (types.int32,)]

def resolve_type(context, op):
    """
    Resolve types for ops:

        - map numba type to low-level representation type
        - represent stack-allocated values through pointers
    """
    if isinstance(op, (FuncArg, Const, Op, Undef)):
        if op.type.is_void:
            return op

        if op not in context:
            raise errors.CompileError("Type for %s was lost" % (op,))

        # Retrieve type
        type = context[op]

        # Remove dummy method lookups (TODO: methods as first-class citizens)
        if type.__class__.__name__ == 'Method':
            return op # TODO: Remove this

        # Generate low-level representation type
        if isinstance(type, set):
            assert not isinstance(type, set)
        ltype = compiler.representation_type(type)

        if isinstance(op, Const):
            const = op.const

            # Represent dummy constant structs with at least one field for LLVM
            if isinstance(const, Struct) and not const.values:
                const = Struct(['dummy'], [Const(0, ptypes.Int32)])

            # Also represent stack-allocated values through pointers
            if ltype.is_pointer and not isinstance(const, Pointer):
                const = Pointer(const, ltype)

            op = Const(const, ltype)
        elif isinstance(op, Undef):
            op = Undef(ltype)
        else:
            op.type = ltype

    return op

#===------------------------------------------------------------------===
# Passes
#===------------------------------------------------------------------===

def lltyping(func, env):
    """Annotate the function with the low-level representation types"""
    if not env['numba.state.opaque']:
        context = env['numba.typing.context']
        resolve = partial(resolve_type, context)

        for arg in func.args:
            resolve(arg)
        for op in func.ops:
            if op.opcode == 'exc_catch':
                continue
            op.replace(resolve(op))
            op.set_args(nestedmap(resolve, op.args))

        restype = env['numba.typing.restype']
        if conversion.byref(restype):
            ll_restype = ptypes.Void
        else:
            ll_restype = compiler.representation_type(restype)

        func.type = ptypes.Function(ll_restype, [arg.type for arg in func.args],
                                    False)
        #signature = env['numba.typing.signature']
        #func.type = ll_type(signature).base


run = lltyping