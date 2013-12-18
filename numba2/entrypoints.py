# -*- coding: utf-8 -*-

"""
Entry points for runtime code.
"""

from __future__ import print_function, division, absolute_import
import sys
import types
from functools import partial, wraps

from .functionwrapper import wrap
from .typing import MetaType
from .utils import applyable, applyable_decorator

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
    kwds['scope'] = kwds.pop('scope', sys._getframe(1).f_locals)

    if isinstance(f, (type, types.FunctionType, types.ClassType)):
        return _jit(f, *args, **kwds)

    arg = f
    return lambda f: _jit(f, arg, *args, **kwds)


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
    if 'scope' not in kwds:
        kwds['scope'] = {} # TODO: retrieve scope for @ijit, @abstract, etc
    return wrap(f, signature, **kwds)


def jit_class(cls, signature=None, abstract=False, stackallocate=False, scope=None):
    """
    @jit('NDArray[dtype, ndim]')
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

def scoping_decorator(decorator):
    @wraps(decorator)
    def decorator_wrapper(*args, **kwargs):
        scope = kwargs.pop('scope', sys._getframe(1).f_locals)
        if applyable(args, kwargs):
            return decorator(*args, scope=scope)
        return lambda f: decorator(f, *args, scope=scope, **kwargs)

    return decorator_wrapper

@scoping_decorator
def abstract(f, *args, **kwds):
    kwds['abstract'] = True
    return _jit(f, *args, **kwds)

# --- shorthands

@scoping_decorator
def ijit(f, *args, **kwds):
    """@jit(inline=True)"""
    return _jit(f, *args, inline=True, **kwds)

@scoping_decorator
def sjit(cls, *args, **kwds):
    """@jit(stackallocate=True)"""
    if hasattr(cls, '__del__'):
        raise TypeError(
            "Cannot stack-allocate instances with __del__: %s" % (cls,))
    return jit_class(cls, *args, stackallocate=True, **kwds)

#ijit = partial(jit, inline=True)
#sjit = partial(jit, stackallocate=True)