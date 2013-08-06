# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys

class Dispatcher(object):
    """Dispatcher for overloaded functions"""
    def __init__(self):
        self.overloads = {}

    def add_overload(self, f, signature):
        # TODO: assert signature is compatible with current signatures
        self.overloads[f] = signature

    def dispatch(self, *args, **kwargs):
        pass

    def __repr__(self):
        f = iter(self.overloads).next()
        return '<%s: %s>' % (f.__name__, list(self.overloads.itervalues()))

def overload(signature, func=None):
    """
    Overload `func` with new signature, or find this function in the local
    scope with the same name.

        @overload('Array[dtype, ndim] -> dtype')
        def myfunc(...):
            ...
    """
    dispatcher = func or sys._getframe(1).f_locals.get(func.__name__)
    dispatcher = dispatcher or Dispatcher()

    def decorator(f):
        dispatcher.add_overload(f, signature)
        return dispatcher

    return decorator

def overloadable():
    """
    Make a function overloadable, useful if there's no useful defaults to
    overload on
    """
    return Dispatcher()
