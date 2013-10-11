# -*- coding: utf-8 -*-

"""
Entry points for runtime code.
"""

from __future__ import print_function, division, absolute_import
import types
from functools import partial

from .functionwrapper import wrap
from .typing import MetaType, set_type_data
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
    from .runtime.classes import allocate_type_constructor, patch_class

    if not abstract and not hasattr(cls, 'layout'):
        raise ValueError("layout of class %s not set" % (cls,))

    constructor, type = allocate_type_constructor(cls, signature)
    cls.type = type
    if not abstract:
        patch_class(cls)

    result = MetaType(cls.__name__, cls.__bases__, dict(vars(cls)))
    set_type_data(constructor, result)
    return result


@applyable_decorator
def abstract(f, *args, **kwds):
    kwds['abstract'] = True
    return _jit(f, *args, **kwds)

# --- shorthands

ijit = partial(jit, inline=True)