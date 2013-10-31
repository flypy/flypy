# -*- coding: utf-8 -*-

"""
Preparation for codegen.
"""

from __future__ import print_function, division, absolute_import
from functools import partial

from numba2 import types, errors
from numba2 import representation
from numba2.runtime import conversion

from pykit.ir import FuncArg, Op, Const, Pointer, Struct
from pykit import types as ptypes
from pykit.utils import nestedmap

#===------------------------------------------------------------------===
# Types
#===------------------------------------------------------------------===

dummy_type = [('dummy',), (types.int32,)]

def ll_type(x):
    """
    Get the low-level representation type for a high-level (user-defined) type.
    """
    return representation.representation_type(x)


def resolve_type(context, op):
    if isinstance(op, (FuncArg, Const, Op)):
        if not op.type.is_void:
            if op not in context:
                raise errors.CompileError("Type for %s was lost" % (op,))
            type = context[op]
            if type.__class__.__name__ == 'Method':
                return op # TODO: Remove this

            ltype = ll_type(type)
            if isinstance(op, Const):
                const = op.const
                if isinstance(const, Struct) and not const.values:
                    const = Struct(['dummy'], [Const(0, ptypes.Int32)])
                if ltype.is_pointer and not isinstance(const, Pointer):
                    const = Pointer(const, ltype)
                op = Const(const, ltype)
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
        if conversion.stack_allocate(restype):
            ll_restype = ptypes.Void
        else:
            ll_restype = ll_type(restype)

        func.type = ptypes.Function(ll_restype, [arg.type for arg in func.args])


run = lltyping