# -*- coding: utf-8 -*-

"""
Preparation for codegen.
"""

from __future__ import print_function, division, absolute_import
from functools import partial
from collections import defaultdict

from numba2 import types, typing

from pykit.ir import FuncArg, Op, Const, Pointer, Struct
from pykit import types as ptypes
from pykit.utils import nestedmap

#===------------------------------------------------------------------===
# Types
#===------------------------------------------------------------------===

def _type_constructor(x):
    """Given a numba class, get the type constructor class"""
    # TODO: This muck should go away once we finish typedefs between python
    # objects and numba objects (e.g. bool <-> Bool)
    return type(x.type)

_typemap = {
    _type_constructor(types.Function)   : ptypes.Function,
    _type_constructor(types.Void)       : ptypes.VoidT,
    _type_constructor(types.Bool)       : ptypes.Boolean,
    _type_constructor(types.Int)        : ptypes.Integral,
    _type_constructor(types.Float)      : ptypes.Real,
    _type_constructor(types.Pointer)    : ptypes.Pointer,
    #_type_constructor(types.Struct)     : ptypes.Struct,
}

dummy_type = [('dummy',), (types.int32,)]

def ll_type(x, seen=None):
    """
    Get the low-level representation type for a high-level (user-defined) type.
    """
    if seen is None:
        seen = defaultdict(int)
    if seen[x]:
        raise NotImplementedError("Recursive types", x)

    # TODO: Implement type resolution in a single pass !
    x = typing.resolve_type(x)
    seen[x] += 1
    if not isinstance(x, types.Type):
        lltype = x
    elif type(x) in _typemap:
        ctor = _typemap[type(x)]
        lltype = ctor(*map(ll_type, x.parameters))
    else:
        fields = x.layout
        names, field_types = zip(*fields.items()) or dummy_type
        field_types = [typing.resolve_simple(x, t) for t in field_types]
        lltype = ptypes.Pointer(ptypes.Struct(
            names, [ll_type(t, seen) for t in field_types]))

    seen[x] -= 1
    return lltype

def resolve_type(context, op):
    if isinstance(op, (FuncArg, Const, Op)):
        if not op.type.is_void:
            type = context[op]
            if type.__class__.__name__ == 'Method':
                return op # TODO: Remove this

            ltype = ll_type(type)
            if isinstance(op, Const):
                const = op.const
                if isinstance(const, Struct) and not const.values:
                    const = Struct(['dummy'], [Const(0, ptypes.Int32)])
                if ltype.is_pointer and not isinstance(const, Pointer):
                    const = Pointer(const)
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
            op.replace(resolve(op))
            op.set_args(nestedmap(resolve, op.args))

        func.type = ptypes.Function(ll_type(env['numba.typing.restype']),
                                    [arg.type for arg in func.args])


run = lltyping