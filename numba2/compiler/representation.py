# -*- coding: utf-8 -*-

"""
Construct numba objects from python values.
"""

from __future__ import print_function, division, absolute_import

from collections import defaultdict

from numba2 import typing
from numba2.types import Mono, Bool, Int, Float, Function, Void, Pointer, int32

from pykit.ir import Const, Struct
from pykit import types as ptypes

#===------------------------------------------------------------------===
# Type Representation
#===------------------------------------------------------------------===

def _type_constructor(x):
    """Given a numba class, get the type constructor class"""
    # TODO: This muck should go away once we finish typedefs between python
    # objects and numba objects (e.g. bool <-> Bool)
    return type(x.type)

_typemap = {
    _type_constructor(Function)   : ptypes.Function,
    _type_constructor(Void)       : ptypes.VoidT,
    _type_constructor(Bool)       : ptypes.Boolean,
    _type_constructor(Int)        : ptypes.Integral,
    _type_constructor(Float)      : ptypes.Real,
    _type_constructor(Pointer)    : ptypes.Pointer,
}

dummy_type = [('dummy',), (int32,)]

def representation_type(x, seen=None):
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

    if not isinstance(x, Mono):
        rep = x
    elif type(x) in _typemap:
        ctor = _typemap[type(x)]
        rep = ctor(*[representation_type(p, seen) for p in x.parameters])
    else:
        # First resolve the layout
        fields = x.layout
        names, field_types = zip(*fields.items()) or dummy_type
        field_types = [typing.resolve_simple(x, t) for t in field_types]

        # Struct representation
        rep = ptypes.Struct(
            names, [representation_type(t, seen) for t in field_types])

        # See whether it needs to be stack allocated
        if not stack_allocate(x):
            rep = ptypes.Pointer(rep)

    seen[x] -= 1
    return rep


def stack_allocate(type):
    """
    Determine whether values of this type should be stack-allocated and partake
    directly as values under composition.
    """
    return False


def build_struct_value(ty, value, seen=None):
    """
    Build a constant struct value from the given runtime Python
    user-defined object.
    """
    seen = seen or set()
    if id(value) in seen:
        raise TypeError("Cannot use recursive value as a numba constant")
    seen.add(id(value))

    names, types = zip(*dict(ty.layout).items()) or [(), ()]
    values = [getattr(value, name) for name in names]
    values = [build_struct_value(ty, value, seen)
                  for ty, value in zip(types, values)]
    return Struct(names, values)