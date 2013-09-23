# -*- coding: utf-8 -*-

"""
Preparation for codegen.
"""

from __future__ import print_function, division, absolute_import

from numba2 import types, typing
from pykit.ir import vmap, GlobalValue
from pykit import types as ptypes

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
}

def ll_type(x):
    """
    Get the low-level representation type for a high-level (user-defined) type.
    """
    x = typing.resolve_type(x)

    if type(x) in _typemap:
        ctor = _typemap[type(x)]
        lltype = ctor(*x.parameters)
    else:
        t = typing.get_type_data(type(x))
        fields = t.layout
        names, types = zip(*fields)
        lltype = ptypes.Struct(names, [ll_type(t) for t in types])

    return lltype

#===------------------------------------------------------------------===
# Passes
#===------------------------------------------------------------------===

def ll_annotate(func, env):
    """Annotate the function with the low-level representation types"""
    def resolve_type(op):
        if not isinstance(op, GlobalValue):
            if not op.type.is_void:
                type = context[op]
                if type.__class__.__name__ == 'Method':
                    return
                ltype = ll_type(type)
                op.type = ltype

    context = env['numba.typing.context']
    print(func)
    vmap(resolve_type, func)
    print(func)
    print('...')

run = ll_annotate