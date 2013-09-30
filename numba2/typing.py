# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import re
import sys

from blaze import datashape as ds
from blaze.datashape import (TypeVar, TypeConstructor, dshape,
                             coerce, unify, unify_simple, free)

#===------------------------------------------------------------------===
# Parsing
#===------------------------------------------------------------------===

def parse(s):
    if  re.match('\w+$', s):
        return TypeConstructor(s, 0, [])
    return dshape(s)

def resolve_type(t):
    from . import types

    _blaze2numba = {
        ds.bool_   : types.bool_,
        ds.int32   : types.int32,
        ds.float64 : types.float64,
    }

    return ds.tmap(lambda x: _blaze2numba.get(x, x), t)

#===------------------------------------------------------------------===
# Runtime
#===------------------------------------------------------------------===

@property
def bound(self):
    freevars = free(self.impl.type)
    # assert len(freevars) == len(key)

    # TODO: Parameterization by type terms
    return dict((t.symbol, v) for t, v in zip(freevars, self.parameters))


class MetaType(type):
    """
    Type of types.

    Attributes:

        layout: [(str, Type)]
            Layout of the type

        fields: {str: FunctionWrapper}
            Dict of methods
    """

    def __init__(self, name, bases, dct):
        if 'type' not in dct:
            return

        type = dct['type']
        self.layout = layout = dict(getattr(self, 'layout', {}))

        # Set method fields
        self.fields = fields = dict(_extract_fields(type, dct))

        # Verify signatures
        #for func in self.fields.itervalues():
        #    verify_method_signature(type, func.signature)

        # Construct layout
        for attr, t in layout.iteritems():
            if isinstance(t, basestring):
                layout[attr] = parse(t)

        # Patch concrete type with fields, layout
        type_constructor = type.__class__
        type_constructor.impl   = self
        type_constructor.fields = fields
        type_constructor.layout = layout
        type_constructor.bound = bound

        modname = dct['__module__']
        type_constructor.scope = vars(sys.modules[modname])

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Construct concrete type
        constructor = type(self.type)
        result = constructor(*key)

        return result

#===------------------------------------------------------------------===
# Utils
#===------------------------------------------------------------------===

def _extract_fields(type, dct):
    from .functionwrapper import FunctionWrapper # circular...
    from . import typing

    fields = {}
    for name, value in dct.iteritems():
        if isinstance(value, FunctionWrapper):
            fields[name] = value

    # TODO: layout...

    return fields

def verify_method_signature(type, signature):
    """Verify a method signature in the context of the defining type"""
    typebound = set([t.symbol for t in free(type)])
    sigbound = set([t.symbol for argtype in signature.argtypes
                                 for t in free(argtype)])
    for t in free(signature.restype):
        if t.symbol not in typebound and t.symbol not in sigbound:
            raise TypeError("Type variable %s is not bound by the type or "
                            "argument types" % (t,))

#===------------------------------------------------------------------===
# Unification and type resolution
#===------------------------------------------------------------------===

def resolve_in_scope(t, scope):
    """
    Resolve a parsed type in the current scope. For example, if we parse
    Foo[X], look up Foo in the current scope and reconstruct it with X.
    """
    def resolve(t):
        if isinstance(type(t), TypeConstructor):
            name = type(t).name
            if name not in scope:
                raise TypeError(
                    "Type constructor %s is not in the current scope")

            # Get the @jit class (e.g. Int)
            impl = scope[name]

            # Get the TypeConstructor for the @jit class (e.g.
            # Int[nbits, unsigned])
            ctor = impl.type.__class__

            return ctor(*t.parameters)

        return t

    return ds.tmap(resolve, t)

def substitute(solution, t):
    """
    Substitute bound parameters for the corresponding free variables
    """
    def f(t):
        if isinstance(t, TypeVar):
            return solution.get(t.symbol, t)
        return t

    return ds.tmap(f, t)


def resolve(type, scope, bound):
    """
    Resolve a parsed numba type in its scope.
    Do this before applying unification.
    """
    type = resolve_in_scope(type, scope)
    type = substitute(bound, type)
    return type

#===------------------------------------------------------------------===
# Registry
#===------------------------------------------------------------------===

class TypedefRegistry(object):
    def __init__(self):
        self.typedefs = {} # builtin -> numba function

    def typedef(self, pyfunc, numbafunc):
        assert pyfunc not in self.typedefs
        self.typedefs[pyfunc] = numbafunc


typedef_registry = TypedefRegistry()
typedef = typedef_registry.typedef

_registry = {}

def get_type_data(t):
    assert isinstance(t, TypeConstructor)
    r = _registry
    return _registry[t]

def set_type_data(t, data):
    _registry[t] = data

