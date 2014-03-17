# -*- coding: utf-8 -*-

"""
Python's standard exception class hierarchy.

Exceptions found here are defined both in the exceptions module and the
built-in namespace.  It is recommended that user-defined exceptions
inherit from Exception.  See the documentation for the exception
inheritance hierarchy.
"""

from __future__ import print_function, division, absolute_import
import builtins as exceptions
from functools import partial

from flypy import overlay, typeof, jit, sjit

__all__ = []

#===------------------------------------------------------------------===
# Decorator
#===------------------------------------------------------------------===

def ejit(py_cls, exc_cls):
    __all__.append(exc_cls.__name__)
    exc_cls = sjit(exc_cls)
    overlay(py_cls, exc_cls)

    @typeof.case(py_cls)
    def exc_typeof(pyval):
        return exc_cls[()]

    return exc_cls

#===------------------------------------------------------------------===
# Exceptions
#===------------------------------------------------------------------===

def jit_hierarchy(cls, cache=None):
    """
    Jit a class hierarchy given by `cls` using the given jit function.

    The resulting classes have an empty layout set.
    """
    if cache is None:
        # Define this for the base case: object does not need to be jitted
        cache = { object: object }

    if cls in cache:
        # Seen this already, reuse result
        return cache[cls]

    # New exception, turn into @jit class
    jitted_bases = tuple(jit_hierarchy(base, cache)
                             for base in cls.__bases__)
    cls_copy = type(cls.__name__,
                    jitted_bases,
                    {'layout': [], '__doc__': cls.__doc__ })
    jitted_cls = ejit(cls, cls_copy)
    cache[cls] = jitted_cls
    return jitted_cls


def define_exceptions():
    """Define all exceptions from the `exceptions` module as jitted classes"""
    result = {}
    cache = { object: object }
    for name, cls in vars(exceptions).items():
        if isinstance(cls, type) and issubclass(cls, (exceptions.BaseException,
                                                      exceptions.Warning)):
            result[name] = jit_hierarchy(cls, cache)

    return result


globals().update(define_exceptions())