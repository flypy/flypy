# -*- coding: utf-8 -*-

"""
Utilities!
"""

from __future__ import print_function, division, absolute_import

import types
import functools
try:
    from collections import MutableMapping
except ImportError as e:
    # Python 3
    from UserDict import DictMixin as MutableMapping

#===------------------------------------------------------------------===
# Decorators
#===------------------------------------------------------------------===

def applyable_decorator(decorator):
    """
    Construct an applyable decorator, that always calls the decorator
    with the function and any optional args and keyword args:

        @applyable_decorator
        def decorator(f, *args, **kwds):
            ...

    Then use `decorator` as follows:

        @decorator
        def foo():
            ...

    or

        @decorator(value='blah')
        def foo():
            ...

    This does not work if the arguments to `decorator` are themselves
    functions!
    """
    @functools.wraps(decorator)
    def decorator_wrapper(*args, **kwargs):
        if len(args) == 1 and not kwargs:
            f = args[0]
            if isinstance(f, (type, types.FunctionType, types.ClassType)):
                from .types import Type
                if not isinstance(f, Type):
                    return decorator(args[0])

        return lambda f: decorator(f, *args, **kwargs)

    return decorator_wrapper

#===------------------------------------------------------------------===
# Properties
#===------------------------------------------------------------------===

class TypedProperty(object):
    '''Defines a class property that does a type check in the setter.'''

    def __new__(cls, ty, doc, default=None):
        rv = super(TypedProperty, cls).__new__(cls)
        cls.__init__(rv, ty, doc, default)
        return property(rv.getter, rv.setter, rv.deleter, doc)

    def __init__(self, ty, doc, default=None):
        self.propname = '_numba_property_%d' % (id(self),)
        self.default = default
        self.ty = ty
        self.doc = doc

    def getter(self, obj):
        return getattr(obj, self.propname, self.default)

    def setter(self, obj, new_val):
        if not isinstance(new_val, self.ty):
            raise ValueError(
                'Invalid property setting, expected instance of type(s) %r '
                '(got %r).' % (self.ty, type(new_val)))
        setattr(obj, self.propname, new_val)

    def deleter(self, obj):
        delattr(obj, self.propname)

#===------------------------------------------------------------------===
# Data Structures
#===------------------------------------------------------------------===

class FrozenDict(MutableMapping):
    """
    Immutable dict.
    """

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        raise ValueError("This dict is immutable")

    def __delitem__(self, key):
        raise ValueError("This dict is immutable")

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def keys(self):
        return list(self.data)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        return cls(dict.fromkeys(iterable, value))
