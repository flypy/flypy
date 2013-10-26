# -*- coding: utf-8 -*-

"""
Entry points for runtime code.
"""

from __future__ import print_function, division, absolute_import
import types
from functools import partial

from .functionwrapper import wrap
from .typing import MetaType
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


def jit_func(f, signature=None, **kwds):
    """
    @jit('a -> List[a] -> List[a]')
    """
    return wrap(f, signature, **kwds)


def jit_class(cls, signature=None, abstract=False, stackallocate=False):
    """
    @jit('Array[dtype, ndim]')
    """
    from .runtime.classes import allocate_type_constructor, patch_class
    from .runtime.interfaces import copy_methods

    if not abstract and not hasattr(cls, 'layout'):
        raise ValueError("layout of class %s not set" % (cls,))

    constructor, type = allocate_type_constructor(cls, signature)
    cls.type = type
    cls.stackallocate = stackallocate
    if not abstract:
        patch_class(cls)

    for base in cls.__mro__:
        copy_methods(cls, base)

    return MetaType(cls.__name__, cls.__bases__, dict(vars(cls)))


@applyable_decorator
def abstract(f, *args, **kwds):
    kwds['abstract'] = True
    return _jit(f, *args, **kwds)

# --- shorthands

@applyable_decorator
def ijit(f, *args, **kwds):
    """@jit(inline=True)"""
    return _jit(f, *args, inline=True, **kwds)

@applyable_decorator
def sjit(cls, *args, **kwds):
    """@jit(stackallocate=True)"""
    return jit_class(cls, *args, stackallocate=True, **kwds)