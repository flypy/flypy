# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import re
import sys

from pykit.utils import hashable

from blaze import datashape as ds
from blaze.datashape import (TypeVar, TypeConstructor, dshape,
                             coercion_cost as coerce, unify, unify_simple, free)

#===------------------------------------------------------------------===
# Parsing
#===------------------------------------------------------------------===

def parse(s):
    if s[0].isupper() and re.match('\w+$', s): # HACK
        return TypeConstructor(s, 0, [])
    return dshape(s)

def typemap():
    from . import types

    _blaze2numba = {
        ds.bool_   : types.bool_,
        ds.int32   : types.int32,
        ds.int64   : types.int64,
        ds.float32 : types.float32,
        ds.float64 : types.float64,
    }
    return _blaze2numba

# TODO: implement our own typing rules

def resolve_type(t):
    _blaze2numba = typemap()
    return ds.tmap(lambda x: _blaze2numba.get(x, x), t)

def to_blaze(t):
    replacements = dict((v, k) for k, v in typemap().iteritems())
    return ds.tmap(lambda x: replacements.get(x, x), t)

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

        layout: {str: Type}
            Layout of the type

        fields: {str: FunctionWrapper}
            Dict of methods
    """

    _is_numba_class = True

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

        @property
        def resolved_layout(self):
            return dict((n, resolve_simple(self, t)) for n, t in layout.items())

        type_constructor.resolved_layout = resolved_layout

        modname = dct['__module__']
        module = sys.modules.get(modname)
        type_constructor.scope = vars(module) if module else {}

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

def lookup_builtin_type(name):
    from . import types

    builtin_scope = {
        'Function': types.Function,
        'Pointer':  types.Pointer,
        'Bool':     types.Bool,
        'Int':      types.Int,
        'Float':    types.Float,
        'Void':     types.Void,
    }

    return builtin_scope.get(name)

def resolve_in_scope(t, scope):
    """
    Resolve a parsed type in the current scope. For example, if we parse
    Foo[X], look up Foo in the current scope and reconstruct it with X.
    """
    def resolve(t):
        if isinstance(type(t), TypeConstructor):
            name = type(t).name

            # Get the @jit class (e.g. Int)
            if hasattr(t, 'impl'):
                impl = t.impl # already resolved!
            else:
                impl = scope.get(name) or lookup_builtin_type(name)

            if impl is None:
                raise TypeError(
                    "Type constructor %r is not in the current scope" % (name,))

            # Get the TypeConstructor for the @jit class (e.g.
            # Int[nbits, unsigned])
            ctor = impl.type.__class__

            return ctor(*t.parameters)

        elif isinstance(t, TypeVar) and t.symbol[0].isupper():
            # Resolve bare types, e.g. a name like 'NoneType' is parsed as a
            # TypeVar
            if t.symbol == 'NoneType':
                assert t.symbol in scope
            if scope.get(t.symbol):
                cls = scope[t.symbol]
                return cls[()]
            return t

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
    type = resolve_type(type)
    type = resolve_in_scope(type, scope)
    type = substitute(bound, type)
    if isinstance(type, ds.DataShape) and not type.shape: # HACK
        type = type.measure
    return type

def resolve_simple(defining_type, type):
    """
    Resolve type `type` with respect to the scope and bound variables of
    `defining_type`.

    E.g. if we have

        class C(object):
            layout = [('x', 'B[int32]')]

    we must resolve B as a class in the scope `C` is defined in.
    """
    return resolve(type, defining_type.scope, defining_type.bound)

#===------------------------------------------------------------------===
# Registry
#===------------------------------------------------------------------===

class OverlayRegistry(object):
    def __init__(self):
        self.overlays = {} # builtin -> numba function

    def overlay(self, pyfunc, numbafunc):
        assert pyfunc not in self.overlays
        self.overlays[pyfunc] = numbafunc

    def lookup_overlay(self, pyfunc):
        if not hashable(pyfunc):
            return None
        return self.overlays.get(pyfunc)


overlay_registry = OverlayRegistry()
overlay = overlay_registry.overlay