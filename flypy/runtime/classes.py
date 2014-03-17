# -*- coding: utf-8 -*-

"""
Entry points for runtime code.
"""

from __future__ import print_function, division, absolute_import
import textwrap

from flypy.typing import parse

#===------------------------------------------------------------------===
# Class signatures
#===------------------------------------------------------------------===

def allocate_type_constructor(cls, signature):
    """Allocate a type constructor for an @jit class"""
    from flypy.coretypes import TypeConstructor

    if signature is not None:
        t, name, params = parse_constructor(signature)
        if name != cls.__name__:
            raise TypeError(
                "Got differing names for type constructor and class, "
                "%s and %s" % (name, cls.__name__))

        constructor = type(t)
        return constructor, t
    else:
        constructor = TypeConstructor(cls.__name__, 0, [])
        return constructor, constructor()

def parse_constructor(signature):
    """Parse a type pass to @jit on a class"""
    from flypy.coretypes import Mono, TypeVar, EllipsisT

    if isinstance(signature, str):
        t = parse(signature)
    else:
        t = signature

    if isinstance(t, TypeVar):
        name = t.symbol
        params = ()
    elif not isinstance(t, Mono):
        raise TypeError(
            "Expected a type variable or type constructor as a signature, got %s" % (t,))
    else:
        name = type(t).__name__
        params = t.parameters

    for i, param in enumerate(params):
        if not isinstance(param, (TypeVar, EllipsisT)):
            raise TypeError(
                "Parameter %s is not a type variable! Got %s." % (i, param))

    return t, name, params

#===------------------------------------------------------------------===
# __allocate__
#===------------------------------------------------------------------===

def patch_class(cls):
    """
    Patch a flypy @jit class with a __init__ if not present.
    """
    from ..entrypoints import ijit

    if '__init__' not in vars(cls):
        names = [name for name, type in cls.layout]
        cls.__init__ = ijit(fabricate_init(names))

def fabricate_init(names):
    stmts = ["self.%s = %s" % (name, name) for name in names] or ["pass"]

    source = textwrap.dedent("""
    def __init__(self, %s):
        %s
    """) % (", ".join(names), "\n    ".join(stmts))

    result = {}
    exec(source, result, result)

    __init__ = result["__init__"]
    del result["__init__"]
    return __init__
