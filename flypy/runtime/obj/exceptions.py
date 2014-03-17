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

results = {}

# Take exceptions from `builtins` and turn them into `@jit` classes
for name, cls in vars(exceptions).items():
    if cls in results:
        # Seen this already, this is an alias: reuse previous result
        flypy_exc = results[cls]
    elif isinstance(cls, type) and issubclass(cls, (exceptions.BaseException,
                                                    exceptions.Warning)):
        # New exception, turn into @jit class
        flypy_exc = type(cls.__name__,
                         cls.__bases__,
                         {'layout': [], '__doc__': cls.__doc__ })
        flypy_exc = ejit(cls, flypy_exc)
        results[cls] = flypy_exc
    else:
        continue

    globals()[name] = flypy_exc
