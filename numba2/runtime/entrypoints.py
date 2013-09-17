# -*- coding: utf-8 -*-

"""
Entry points for runtime code.
"""

from __future__ import print_function, division, absolute_import
import types
from functools import partial

from .types import Type
from ..compiler import annotate
from ..utils import applyable_decorator

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
    if isinstance(f, (types.ClassType, type)):
        return jit_class(f, *args, **kwds)
    else:
        assert isinstance(f, types.FunctionType)
        return jit_func(f, *args, **kwds)


@applyable_decorator
def jit_func(f, signature=None):
    """
    @jit('a -> List[a] -> List[a]')
    """
    if not signature:
        raise ValueError("Require signature") # TODO: fabricate generic signature
    annotate(f, type_signature=signature)
    return f


@applyable_decorator
def jit_class(cls, signature=None):
    """
    @jit('Array[dtype, ndim]')
    """
    if not hasattr(cls, 'layout'):
        raise ValueError("layout of class %s not set" % (cls,))

    assert not hasattr(cls, 'parameters')
    assert not hasattr(cls, 'type')

    # Type.register(cls)

    if signature is not None:
        type = parse_type(signature)
        cls.parameters = free(type)
        cls.layout = substitute(cls.layout, )
    else:
        cls.parameters = ()
        assert not free(cls.layout)

    return Type(cls.__name__, cls.__bases__, vars(cls))


@applyable_decorator
def abstract(f, *args, **kwds):
    kwds['abstract'] = True
    return jit(f, *args, **kwds)


# --- shorthands

ijit = partial(jit, inline=True)