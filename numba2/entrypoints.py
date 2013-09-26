# -*- coding: utf-8 -*-

"""
Entry points for runtime code.
"""

from __future__ import print_function, division, absolute_import
import types
from functools import partial

from .functionwrapper import wrap
from .typing import MetaType, parse, set_type_data
from .utils import applyable_decorator

@applyable_decorator
def jit(f, *args, **kwds):
    """
    @jit entry point:

        @jit
        def myfunc(a, b): return a + b

        @jit('a -> b')
        def myfunc(a, b): return a + b

        @jit
        class Foo(object): pass

        @jit('Foo[a]')
        class Foo(object): pass
    """
    return _jit(f, *args, **kwds)

def _jit(f, *args, **kwds):
    if isinstance(f, (types.ClassType, type)):
        return jit_class(f, *args, **kwds)
    else:
        assert isinstance(f, types.FunctionType)
        return jit_func(f, *args, **kwds)


def jit_func(f, signature=None, abstract=False, opaque=False):
    """
    @jit('a -> List[a] -> List[a]')
    """
    return wrap(f, signature, abstract=abstract, opaque=opaque)


def jit_class(cls, signature=None, abstract=False):
    """
    @jit('Array[dtype, ndim]')
    """
    from .types import free, TypeConstructor

    if not abstract and not hasattr(cls, 'layout'):
        raise ValueError("layout of class %s not set" % (cls,))

    dct = dict(vars(cls))

    if signature is not None:
        t, name, params = parse_constructor(signature)
        if name != cls.__name__:
            raise TypeError(
                "Got differing names for type constructor and class, "
                "%s and %s" % (name, cls.__name__))
        constructor = type(t)
        dct['type'] = t
    else:
        constructor = TypeConstructor(cls.__name__, 0, [])
        dct['type'] = constructor()
        if not abstract:
            assert not free(cls.layout)

    result = MetaType(cls.__name__, cls.__bases__, dct)
    set_type_data(constructor, result)
    return result


def parse_constructor(signature):
    from .types import Type, TypeVar

    if isinstance(signature, basestring):
        t = parse(signature)
    else:
        t = signature

    if isinstance(t, TypeVar):
        name = t.symbol
        params = ()
    elif not isinstance(t, Type):
        raise TypeError(
            "Expected a type variable or type constructor as a signature, got %s" % (t,))
    else:
        name = type(t).__name__
        params = t.parameters

    for i, param in enumerate(params):
        if not isinstance(param, TypeVar):
            raise TypeError(
                "Parameter %s is not a type variable! Got %s." % (i, param))

    return t, name, params

@applyable_decorator
def abstract(f, *args, **kwds):
    kwds['abstract'] = True
    return _jit(f, *args, **kwds)

# --- shorthands

ijit = partial(jit, inline=True)