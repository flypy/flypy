# -*- coding: utf-8 -*-

"""
Utilities!
"""

from __future__ import print_function, division, absolute_import

import types
import functools

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
        if (len(args) == 1 and not kwargs and
                isinstance(args[0], types.FunctionType)):
            return decorator(args[0])
        else:
            return lambda f: decorator(f, *args, **kwargs)

    return decorator_wrapper


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
